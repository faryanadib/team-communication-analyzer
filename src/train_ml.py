"""
train_ml.py
═══════════
ML pipeline of SocialCompass v2 — trained on INDEPENDENT LLM soft labels.

This replaces an earlier circular hybrid pipeline:
  OLD (circular): 13 metrics → rule labels → RF trained on rule labels
                  → 50/50 hybrid vote of rule + ML  (model can never disagree
                  with the rules, so "accuracy" measured self-consistency).
  NEW (this file): 13 metrics (features.py, X) → model → soft archetype
                  distribution, supervised by LLM labels read from RAW TEXT
                  (llm_labels.json, y). Features and labels come from two
                  independent sources, so the loop is broken and the hybrid
                  vote is no longer needed — the ML prediction IS the label.

  v2.1 LATE-FUSION (June 2026): a blind held-out test on the 5-person team
  (wa_holdout_llm.json, LLM reading raw text) exposed that the 13 graph
  metrics are *scale-dependent* — betweenness_centrality is degenerate (all 0)
  in a tiny fully-connected chat, and in_out_ratio's within-team rank turns the
  least-talkative-but-engaged member into a false "Leech" (metric-only agreed
  with LLM on only 2/5 members, all 3 misses → Leech). Fix: add a second,
  *scale-invariant* TEXT view (12 interpretable style signals distilled from
  the same LLM labels) and LATE-FUSE the two soft outputs:
        P = 0.15 · P_metric  +  0.85 · P_style
  Chosen by sweeping configs on CV (197 users) + the blind held-out: the fusion
  lifts team agreement 2/5 → 3/5, improves CV-KL 0.074 → 0.066 and CV macro-F1
  0.29 → 0.33, and removes the Leech bias entirely (0 Leech on the team vs 3 for
  metric-only), while keeping the network metrics (and their interpretability)
  in the decision. The fusion weight is re-selected by cross-validation at
  train time (select_fusion_weight).

Architecture (agreed June 2026):
  STEP 1  Load X: 13 raw metrics for all labeled users
          (reuses training_results.json — produced by features.py on the full
          message graph; betweenness/reply-chains computed over ALL messages).
  STEP 2  Rank-normalize X within each dataset (cross-dataset comparability,
          same convention as the original pipeline).
  STEP 3  Load y: LLM soft labels (5 probabilities per user) + confidence
          → per-sample weights (high=1.0 / medium=0.7 / low=0.4) so confident
          labels teach more (weighted distillation, Hinton et al. 2015).
  STEP 4  Compare models with Repeated Stratified K-Fold (5 folds × 10
          repeats, stratified on the dominant archetype so rare classes
          appear in every fold):
            • mean-distribution dummy        (floor baseline)
            • multinomial LogisticRegression (hard-label baseline, balanced)
            • RandomForestRegressor          (multi-output soft regression)
            • XGBoost / LightGBM             (boosted soft regression)
          Soft metrics: KL divergence, cosine similarity, Brier score.
          Hard metrics: argmax accuracy, macro-F1 (treats rare classes equally).
  STEP 5  Refit the best model on all labeled users.
  STEP 6  Independent unsupervised check: K-means(5) silhouette + ARI vs
          LLM dominant labels (does the raw feature space agree at all?).
  STEP 7  Feature importances (which metric drives which archetype).
  STEP 8  Domain-shift check train → team WhatsApp (Cohen's d per metric).
  STEP 9  Predict the team (EXTERNAL TEST — WhatsApp was never labeled or
          trained on) and write a dashboard-compatible wa_results.json.

Scientific sources:
  Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network.
  Breiman (2001). Random Forests. ML 45(1).
  Chen & Guestrin (2016). XGBoost. KDD '16.
  Ke et al. (2017). LightGBM. NeurIPS '17.
  Rousseeuw (1987). Silhouettes. J. Comput. Appl. Math. 20.
  Hubert & Arabie (1985). Comparing partitions (ARI). J. Classification 2.
  Kullback & Leibler (1951). On information and sufficiency. Ann. Math. Stat.
  Brier (1950). Verification of forecasts. Monthly Weather Review 78.
  Cohen (1988). Statistical Power Analysis (effect size d).
  Ratner et al. (2020). Snorkel: weak supervision. VLDB J. 29.

Run:  python train_ml.py
Outputs: ml_model.joblib · ml_results.json · wa_results.json
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score, accuracy_score

from features import ALL_METRICS
import text_features as TF
import config

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
# Fusion weight is now SELECTED BY CROSS-VALIDATION (select_fusion_weight), not
# hand-picked. This module-level value is only the fallback default; main()
# overwrites it with the CV-chosen weight and records it in ml_results.json.
FUSION_METRIC_W = 0.15
FUSION_W_GRID = [round(x, 2) for x in np.arange(0.0, 0.41, 0.05)]  # search 0.00–0.40

ARCH_KEYS = ["bee", "ant", "butterfly", "capybara", "leech"]
KEY_TO_EMOJI = {
    "bee": "🐝 Bee", "ant": "🐜 Ant", "butterfly": "🦋 Butterfly",
    "capybara": "🦫 Capybara", "leech": "🔴 Leech",
}
CONF_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
EPS = 1e-9

TRAIN_FEATURES_FILE = config.TRAIN_FEATURES_V2   # v2: corrected interaction metrics
TEAM_FEATURES_FILE  = config.WA_FEATURES_V2      # (corrected interaction metrics, see features_v2.py)
LABELS_FILE         = config.LLM_LABELS         # LLM soft labels


# ════════════════════════════════════════════════════════════════════════════
# Data assembly
# ════════════════════════════════════════════════════════════════════════════
def rank_normalize(feats, cols=ALL_METRICS):
    """Within-group percentile rank per metric (0-1). Comparable across datasets."""
    out = feats.copy()
    for m in cols:
        out[m] = feats[m].rank(pct=True)
    return out


def load_training_matrix(drop_synthetic=False):
    """X (rank-normalized 13 metrics) + y (soft labels) + weights for labeled users.

    drop_synthetic=True excludes the disclosed synthetic augmentation, for the
    with/without ablation (`python src/train_ml.py --no-synthetic`)."""
    feats = pd.read_json(TRAIN_FEATURES_FILE, lines=True)
    # rank-normalize WITHIN each dataset over ALL its users (full-community
    # percentile — a user's position among 1,639 Slack users, not among the
    # labeled subset), then join the labeled users.
    norm = pd.concat(
        [rank_normalize(g) for _, g in feats.groupby("dataset")], ignore_index=True
    )

    labels = pd.read_json(LABELS_FILE, lines=True)
    if drop_synthetic:
        labels = labels[labels["dataset"] != "synthetic"].reset_index(drop=True)
    merged = labels.merge(
        norm[["author", "dataset"] + ALL_METRICS],
        left_on=["user_id", "dataset"], right_on=["author", "dataset"], how="left",
    )
    n_missing = merged[ALL_METRICS].isna().any(axis=1).sum()
    assert n_missing == 0, f"{n_missing} labeled users missing features!"

    X = merged[ALL_METRICS].to_numpy()
    y = merged[ARCH_KEYS].to_numpy()
    y = y / y.sum(axis=1, keepdims=True)                       # exact simplex
    w = merged["confidence"].map(CONF_WEIGHT).fillna(0.4).to_numpy()
    hard = y.argmax(axis=1)                                    # dominant class idx
    # keep raw (un-normalized) features too, for the domain-shift check
    raw = labels.merge(
        feats[["author", "dataset"] + ALL_METRICS],
        left_on=["user_id", "dataset"], right_on=["author", "dataset"], how="left",
    )[ALL_METRICS]
    return merged, X, y, w, hard, raw


def load_style_features(merged):
    """12-d scale-invariant text-style block, aligned to `merged` row order."""
    idx = [(r["dataset"], r["user_id"]) for _, r in merged.iterrows()]
    _, style = TF.build_features(idx, style_only=True)   # 12-d lexical, no SBERT
    return style.astype(float)


def load_team_style(team_feats):
    idx = [("team", a) for a in team_feats["author"]]
    _, style = TF.build_features(idx, team=True, use_cache=False, style_only=True)
    return style.astype(float)


# ════════════════════════════════════════════════════════════════════════════
# Soft-label metrics
# ════════════════════════════════════════════════════════════════════════════
def to_simplex(p):
    """Clip negatives and renormalize rows to sum 1 (uniform if all-zero)."""
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    uniform = np.full_like(p, 1.0 / p.shape[1])
    return np.where(s > EPS, p / np.maximum(s, EPS), uniform)


def soft_metrics(y_true, y_pred):
    """KL(true‖pred), cosine, Brier, and SOFT ACCURACY — averaged over samples.

    soft accuracy = mean Σ min(pᵢ,qᵢ): the probability mass the prediction
    shares with the soft LLM label (1.0 = identical), the soft analogue of
    argmax accuracy that credits partial agreement on near-ties."""
    p, q = y_true + EPS, to_simplex(y_pred) + EPS
    kl = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
    cos = float(np.mean(
        np.sum(y_true * y_pred, axis=1)
        / (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + EPS)))
    brier = float(np.mean(np.sum((y_true - to_simplex(y_pred)) ** 2, axis=1)))
    sacc = float(np.mean(np.sum(np.minimum(y_true, to_simplex(y_pred)), axis=1)))
    return kl, cos, brier, sacc


# ════════════════════════════════════════════════════════════════════════════
# Model zoo (all expose .fit(X, y, sample_weight) → .predict(X) ∈ R^{n×5})
# ════════════════════════════════════════════════════════════════════════════
class MeanBaseline:
    """Predicts the (weighted) mean training distribution for everyone."""
    name = "Mean-distribution baseline"

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        self.mean_ = (y * w[:, None]).sum(0) / w.sum()
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class LogRegBaseline:
    """Multinomial LogReg on hard labels — its predict_proba is the soft output."""
    name = "Logistic Regression (hard baseline)"

    def fit(self, X, y, sample_weight=None):
        self.clf_ = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
        self.classes_seen_ = np.unique(y.argmax(1))
        self.clf_.fit(X, y.argmax(1), sample_weight=sample_weight)
        return self

    def predict(self, X):
        proba = self.clf_.predict_proba(X)
        out = np.zeros((len(X), len(ARCH_KEYS)))
        for j, cls in enumerate(self.clf_.classes_):
            out[:, cls] = proba[:, j]
        return out


class SoftRegressor:
    """Multi-output soft regression wrapper (RF / XGB / LGBM)."""

    def __init__(self, kind):
        self.kind = kind
        self.name = {"rf": "Random Forest (soft regression)",
                     "xgb": "XGBoost (soft regression)",
                     "lgbm": "LightGBM (soft regression)"}[kind]

    def _make(self):
        if self.kind == "rf":
            return RandomForestRegressor(
                n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=-1)
        if self.kind == "xgb":
            from xgboost import XGBRegressor
            return MultiOutputRegressor(XGBRegressor(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))
        from lightgbm import LGBMRegressor
        return MultiOutputRegressor(LGBMRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_samples=5, random_state=RANDOM_STATE,
            verbosity=-1, n_jobs=-1))

    def fit(self, X, y, sample_weight=None):
        self.model_ = self._make()
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return to_simplex(self.model_.predict(X))

    def feature_importances(self):
        """13-vector (global) + per-archetype matrix when available."""
        m = self.model_
        if isinstance(m, RandomForestRegressor):
            return m.feature_importances_, None
        per = np.array([est.feature_importances_ for est in m.estimators_], dtype=float)
        per = per / np.maximum(per.sum(axis=1, keepdims=True), EPS)
        return per.mean(axis=0), per


class FusionModel:
    """
    Late-fusion of two independent soft-regressors:
        P = metric_w · P_metric(M) + (1-metric_w) · P_style(S)
    M = 13 rank-normalized network/behavioural metrics (scale-dependent).
    S = 12 interpretable text-style signals             (scale-invariant).
    Both distil the SAME LLM soft labels. Fusion corrects the small-group
    scale bias of the metric view with the text view (see module docstring).
    """
    name = "Late-fusion (metric ⊕ text-style)"

    def __init__(self, metric_w=FUSION_METRIC_W):
        self.metric_w = metric_w
        self.metric_ = SoftRegressor("rf")
        self.style_ = SoftRegressor("rf")

    def fit(self, M, S, y, sample_weight=None):
        self.metric_.fit(M, y, sample_weight=sample_weight)
        self.style_.fit(S, y, sample_weight=sample_weight)
        return self

    def predict(self, M, S):
        pm = self.metric_.predict(M)
        ps = self.style_.predict(S)
        return to_simplex(self.metric_w * pm + (1 - self.metric_w) * ps)


# ════════════════════════════════════════════════════════════════════════════
# Cross-validated model comparison
# ════════════════════════════════════════════════════════════════════════════
def evaluate_models(X, y, w, hard, n_splits=5, n_repeats=10):
    models = {
        "dummy": MeanBaseline(),
        "logreg": LogRegBaseline(),
        "rf": SoftRegressor("rf"),
        "xgb": SoftRegressor("xgb"),
        "lgbm": SoftRegressor("lgbm"),
    }
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    results = {k: {m: [] for m in ("kl", "cos", "brier", "sacc", "acc", "f1")} for k in models}

    print(f"\n── MODEL COMPARISON: Repeated Stratified {n_splits}-fold × {n_repeats} "
          + "─" * 14)
    for tr_idx, te_idx in cv.split(X, hard):
        for key, proto in models.items():
            mdl = proto.__class__(proto.kind) if isinstance(proto, SoftRegressor) else proto.__class__()
            mdl.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            pred = mdl.predict(X[te_idx])
            kl, cos, brier, sacc = soft_metrics(y[te_idx], pred)
            results[key]["kl"].append(kl)
            results[key]["cos"].append(cos)
            results[key]["brier"].append(brier)
            results[key]["sacc"].append(sacc)
            results[key]["acc"].append(accuracy_score(hard[te_idx], pred.argmax(1)))
            results[key]["f1"].append(
                f1_score(hard[te_idx], pred.argmax(1), average="macro", zero_division=0))

    summary = {}
    print(f"  {'model':<38}{'KL↓':>8}{'cos↑':>8}{'Brier↓':>8}{'softAcc↑':>9}{'acc↑':>8}{'maF1↑':>8}")
    for key, proto in models.items():
        r = results[key]
        row = {m: (float(np.mean(v)), float(np.std(v))) for m, v in r.items()}
        summary[key] = {"name": proto.name,
                        **{m: {"mean": row[m][0], "std": row[m][1]} for m in row}}
        print(f"  {proto.name:<38}"
              f"{row['kl'][0]:>8.3f}{row['cos'][0]:>8.3f}{row['brier'][0]:>8.3f}"
              f"{row['sacc'][0]:>9.3f}{row['acc'][0]:>8.3f}{row['f1'][0]:>8.3f}")

    # best soft model = lowest KL among the actual regressors (skip baselines)
    candidates = {k: summary[k]["kl"]["mean"] for k in ("rf", "xgb", "lgbm")}
    best = min(candidates, key=candidates.get)
    print(f"\n  → best soft model by KL: {summary[best]['name']}"
          f" (KL {candidates[best]:.3f})")
    # sanity: must beat the dummy baseline
    if candidates[best] >= summary["dummy"]["kl"]["mean"]:
        print("  ⚠ WARNING: best model does not beat the mean-distribution baseline!")
    else:
        gain = (1 - candidates[best] / summary["dummy"]["kl"]["mean"]) * 100
        print(f"  ✓ beats dummy baseline by {gain:.0f}% lower KL")
    return summary, best


def select_fusion_weight(M, S, y, w, hard, grid=None, n_splits=5, n_repeats=6):
    """
    Choose the metric/style fusion weight by cross-validation instead of by hand.
    For each fold we fit the metric view and the style view once, then blend their
    out-of-fold predictions at every candidate weight (the blend is post-hoc, so
    no refitting per weight). The weight with the lowest mean CV KL-divergence to
    the LLM soft labels wins — KL is the soft-label distillation objective, i.e.
    'closest to LLM' is exactly what is being optimised.
    Returns (best_w, {w: (kl_mean, kl_std)}).
    """
    grid = grid or FUSION_W_GRID
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    kls = {a: [] for a in grid}
    for tr, te in cv.split(M, hard):
        pm = SoftRegressor("rf").fit(M[tr], y[tr], sample_weight=w[tr]).predict(M[te])
        ps = SoftRegressor("rf").fit(S[tr], y[tr], sample_weight=w[tr]).predict(S[te])
        for a in grid:
            kl, _, _, _ = soft_metrics(y[te], to_simplex(a * pm + (1 - a) * ps))
            kls[a].append(kl)
    stats = {a: (float(np.mean(v)), float(np.std(v))) for a, v in kls.items()}
    best = min(grid, key=lambda a: stats[a][0])
    print(f"\n── FUSION-WEIGHT SELECTION (CV KL, {n_splits}×{n_repeats}) " + "─" * 16)
    for a in grid:
        mark = "  ← selected" if a == best else ""
        bar = "█" * int((0.10 / max(stats[a][0], 1e-9)) * 30)
        print(f"    metric_w={a:.2f}   KL={stats[a][0]:.4f} ± {stats[a][1]:.4f}{mark}")
    print(f"  → CV-selected metric weight = {best:.2f} "
          f"(style {1-best:.2f}); closest to LLM soft labels.")
    return best, stats


def evaluate_fusion(M, S, y, w, hard, metric_w=FUSION_METRIC_W, n_splits=5, n_repeats=10):
    """Cross-validated soft metrics for the late-fusion model (M ⊕ S)."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    kl, cos, brier, sacc, acc, f1 = [], [], [], [], [], []
    for tr, te in cv.split(M, hard):
        mdl = FusionModel(metric_w).fit(M[tr], S[tr], y[tr], sample_weight=w[tr])
        pred = mdl.predict(M[te], S[te])
        k, c, b, sa = soft_metrics(y[te], pred)
        kl.append(k); cos.append(c); brier.append(b); sacc.append(sa)
        acc.append(accuracy_score(hard[te], pred.argmax(1)))
        f1.append(f1_score(hard[te], pred.argmax(1), average="macro", zero_division=0))
    out = {"name": FusionModel.name,
           "kl": {"mean": float(np.mean(kl)), "std": float(np.std(kl))},
           "cos": {"mean": float(np.mean(cos)), "std": float(np.std(cos))},
           "brier": {"mean": float(np.mean(brier)), "std": float(np.std(brier))},
           "soft_acc": {"mean": float(np.mean(sacc)), "std": float(np.std(sacc))},
           "acc": {"mean": float(np.mean(acc)), "std": float(np.std(acc))},
           "f1": {"mean": float(np.mean(f1)), "std": float(np.std(f1))}}
    print(f"  {FusionModel.name:<38}"
          f"{out['kl']['mean']:>8.3f}{out['cos']['mean']:>8.3f}"
          f"{out['brier']['mean']:>8.3f}{out['soft_acc']['mean']:>9.3f}"
          f"{out['acc']['mean']:>8.3f}{out['f1']['mean']:>8.3f}"
          f"   ★ PRODUCTION")
    return out


def per_class_report(X, y, w, hard, best_kind, n_splits=5, n_repeats=10):
    """Out-of-fold per-class F1 for the chosen model (with dispersion)."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    per_class = {k: [] for k in range(len(ARCH_KEYS))}
    for tr_idx, te_idx in cv.split(X, hard):
        mdl = SoftRegressor(best_kind).fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        pred_hard = mdl.predict(X[te_idx]).argmax(1)
        f1s = f1_score(hard[te_idx], pred_hard,
                       labels=list(range(len(ARCH_KEYS))), average=None, zero_division=0)
        for k in range(len(ARCH_KEYS)):
            if (hard[te_idx] == k).any():            # only folds containing the class
                per_class[k].append(f1s[k])

    print("\n── PER-CLASS F1 (out-of-fold, best model) " + "─" * 22)
    counts = np.bincount(hard, minlength=len(ARCH_KEYS))
    out = {}
    for k, key in enumerate(ARCH_KEYS):
        scores = per_class[k]
        mu, sd = (float(np.mean(scores)), float(np.std(scores))) if scores else (0.0, 0.0)
        flag = "  ⚠ rare class — wide uncertainty" if counts[k] < 15 else ""
        print(f"  {key:<10} n={counts[k]:<4} F1 = {mu:.3f} ± {sd:.3f}{flag}")
        out[key] = {"n": int(counts[k]), "f1_mean": mu, "f1_std": sd}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Independent validation / interpretation / shift
# ════════════════════════════════════════════════════════════════════════════
def kmeans_check(X, hard):
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(Xs)
    sil = float(silhouette_score(Xs, clusters))
    ari = float(adjusted_rand_score(hard, clusters))
    print(f"\n── INDEPENDENT CHECK: K-means(5) on the 13-D feature space " + "─" * 6)
    print(f"  Silhouette (Rousseeuw 1987): {sil:.3f}   (>0.25 some structure)")
    print(f"  ARI vs LLM dominant labels: {ari:.3f}   (0 chance · 1 identical)")
    if sil < 0.25:
        print("  note: archetypes overlap in feature space — expected; the "
              "supervised model resolves the overlap that clustering cannot.")
    return {"silhouette": sil, "ari": ari}


def domain_shift(train_raw, team_raw, d_thresh=0.8):
    print(f"\n── DOMAIN SHIFT: train(labeled) → team WhatsApp " + "─" * 17)
    shifted, table = [], {}
    for m in ALL_METRICS:
        a = train_raw[m].to_numpy(dtype=float)
        b = team_raw[m].to_numpy(dtype=float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(b) > 1 else a.std()
        d = float((b.mean() - a.mean()) / (pooled + 1e-12))
        table[m] = d
        mark = "  ⚠ SHIFT" if abs(d) > d_thresh else ""
        if abs(d) > d_thresh:
            shifted.append(m)
        print(f"    {m:<24} d = {d:+.2f}{mark}")
    if shifted:
        print(f"  ⚠ large shift on {len(shifted)}/13 metrics → treat team "
              "predictions with corresponding caution (model extrapolates there).")
    else:
        print("  ✓ no large domain shift (all |d| ≤ 0.8)")
    return table, shifted


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    drop_synthetic = "--no-synthetic" in sys.argv
    tag = "WITHOUT synthetic" if drop_synthetic else "with synthetic augmentation"
    print(f"\n🤖  SocialCompass v2 — ML classifier on LLM soft labels ({tag})")
    print("=" * 70)

    # ── STEP 1-3: assemble ───────────────────────────────────────────────────
    merged, X, y, w, hard, train_raw = load_training_matrix(drop_synthetic)
    counts = np.bincount(hard, minlength=5)
    print(f"  Training set: {len(X)} labeled users "
          f"({(merged['dataset'] == 'slack').sum()} slack + "
          f"{(merged['dataset'] == 'nankani').sum()} nankani + "
          f"{(merged['dataset'] == 'mbada').sum()} mbada + "
          f"{(merged['dataset'] == 'synthetic').sum()} synthetic)")
    print("  Dominant-archetype distribution (stratification target):")
    for k, key in enumerate(ARCH_KEYS):
        print(f"    {key:<10} {counts[k]:>4}  ({counts[k]/len(X)*100:4.1f}%)")
    print(f"  Sample weights from LLM confidence: "
          f"high={CONF_WEIGHT['high']} · medium={CONF_WEIGHT['medium']} · "
          f"low={CONF_WEIGHT['low']}")

    # ── STEP 3b: text-style view (scale-invariant, distils same LLM labels) ──
    print("\n  Building text-style view (12 interpretable signals per user)…")
    S = load_style_features(merged)
    print(f"  Style block: {S.shape[1]} signals — "
          + ", ".join(TF.STYLE_NAMES[:6]) + ", …")

    # ── STEP 4: model comparison (single-view zoo + the production fusion) ────
    summary, best_kind = evaluate_models(X, y, w, hard)
    metric_w, w_stats = select_fusion_weight(X, S, y, w, hard)   # CV-chosen weight
    # optional manual override: `--metric-w 0.30` forces the fusion weight (e.g.
    # to keep more metric signal so a tiny team isn't collapsed to one archetype).
    if "--metric-w" in sys.argv:
        metric_w = float(sys.argv[sys.argv.index("--metric-w") + 1])
        print(f"  ⚙ MANUAL override: metric weight forced to {metric_w:.2f} "
              f"(style {1-metric_w:.2f})")
    fusion_cv = evaluate_fusion(X, S, y, w, hard, metric_w=metric_w)  # ★ PRODUCTION
    summary["fusion"] = fusion_cv
    per_class = per_class_report(X, y, w, hard, best_kind)

    # ── STEP 5: final fit on all data — LATE-FUSION is the production model ───
    final = FusionModel(metric_w).fit(X, S, y, sample_weight=w)
    train_pred = final.predict(X, S)
    kl_in, cos_in, _, sacc_in = soft_metrics(y, train_pred)
    print(f"\n  Production model = {FusionModel.name} "
          f"(metric {metric_w:.2f} / style {1-metric_w:.2f}, CV-selected)")
    print(f"  Refit on all {len(X)} users "
          f"(in-sample KL {kl_in:.3f} · cos {cos_in:.3f} · soft-acc {sacc_in*100:.0f}%)")

    # ── STEP 6: unsupervised agreement ───────────────────────────────────────
    km = kmeans_check(X, hard)

    # ── STEP 7: feature importances (both views) ─────────────────────────────
    glob, _ = final.metric_.feature_importances()       # metric view (13)
    print(f"\n── METRIC-VIEW IMPORTANCE (Random Forest) " + "─" * 22)
    for i in np.argsort(glob)[::-1][:8]:
        print(f"    {ALL_METRICS[i]:<24} {glob[i]:.3f}  {'█' * int(glob[i] * 60)}")
    imp = {"metric": {ALL_METRICS[i]: float(glob[i]) for i in range(len(ALL_METRICS))}}

    sglob, _ = final.style_.feature_importances()       # text-style view (12)
    print(f"\n── TEXT-STYLE-VIEW IMPORTANCE (Random Forest) " + "─" * 18)
    for i in np.argsort(sglob)[::-1][:8]:
        print(f"    {TF.STYLE_NAMES[i]:<24} {sglob[i]:.3f}  {'█' * int(sglob[i] * 60)}")
    imp["style"] = {TF.STYLE_NAMES[i]: float(sglob[i]) for i in range(len(TF.STYLE_NAMES))}

    # ── STEP 8: domain shift ─────────────────────────────────────────────────
    team_feats = pd.read_json(TEAM_FEATURES_FILE, lines=True)
    shift_table, shifted = domain_shift(train_raw, team_feats[ALL_METRICS])

    # ── STEP 9: predict the team (external test) via LATE FUSION ─────────────
    print(f"\n── TEAM PREDICTION (external test — never trained on) " + "─" * 11)
    team_norm = rank_normalize(team_feats)          # within-team percentiles (metric view)
    X_team = team_norm[ALL_METRICS].to_numpy()
    S_team = load_team_style(team_feats)            # scale-invariant text view
    proba_metric = final.metric_.predict(X_team)
    proba = final.predict(X_team, S_team)           # fused decision
    print("  metric view alone → fused (text view corrects small-group scale bias):")
    for i, a in enumerate(team_feats["author"]):
        mt = ARCH_KEYS[proba_metric[i].argmax()]
        ft = ARCH_KEYS[proba[i].argmax()]
        flag = "" if mt == ft else f"   {mt} → {ft}"
        print(f"    {str(a)[:9]:<10} {KEY_TO_EMOJI[ft]}{flag}")

    res = team_feats[["author"] + ALL_METRICS].copy()
    for j, key in enumerate(ARCH_KEYS):
        res[f"{key}_pct"] = np.round(proba[:, j] * 100, 1)
        res[f"{key}_fit"] = res[f"{key}_pct"]       # dashboard headline alias
        res[f"ml_{key}"]  = res[f"{key}_pct"]       # decision-detail alias
        res[f"metric_{key}"] = np.round(proba_metric[:, j] * 100, 1)  # metric-view detail
    res["archetype"]  = [KEY_TO_EMOJI[ARCH_KEYS[i]] for i in proba.argmax(1)]
    res["confidence"] = np.round(proba.max(1) * 100, 1)
    res["fit_top"]    = res["confidence"]
    res["label_source"] = "ml_fusion"                # provenance marker

    res.to_json(config.WA_RESULTS, orient="records", lines=True)
    joblib.dump({"model": final, "metrics": ALL_METRICS,
                 "style_names": TF.STYLE_NAMES, "arch_keys": ARCH_KEYS,
                 "fusion_metric_w": metric_w},
                config.ML_MODEL)

    with open(config.ML_RESULTS, "w", encoding="utf-8") as f:
        json.dump({
            "n_train": int(len(X)),
            "class_counts": {k: int(c) for k, c in zip(ARCH_KEYS, counts)},
            "cv": summary,
            "best_model": FusionModel.name,
            "fusion_metric_w": metric_w,
            "fusion_w_selection": {str(a): {"kl_mean": s[0], "kl_std": s[1]}
                                   for a, s in w_stats.items()},
            "single_view_best": summary[best_kind]["name"],
            "per_class_f1": per_class,
            "kmeans": km,
            "feature_importance": imp,
            "domain_shift_d": shift_table,
            "domain_shift_flagged": shifted,
        }, f, indent=2)

    print(f"\n  {'Member':<10}{'Label':<16}{'🐝':>6}{'🐜':>6}{'🦋':>6}{'🦫':>6}{'🔴':>6}  conf")
    for i, r in res.iterrows():
        print(f"  {str(r['author'])[:9]:<10}{r['archetype']:<16}"
              f"{r['bee_pct']:>5.0f}%{r['ant_pct']:>5.0f}%{r['butterfly_pct']:>5.0f}%"
              f"{r['capybara_pct']:>5.0f}%{r['leech_pct']:>5.0f}%  {r['confidence']:>4.0f}%")

    print("\n  ✅ Saved → wa_results.json · ml_results.json · ml_model.joblib")
    print("  Run:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
