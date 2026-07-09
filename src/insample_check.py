"""
insample_check.py — Sanity check: run the saved production model on its own
training data.

This is a fit check, not a generalization estimate. The point is the *gap*
between in-sample and the blind held-out score (compare_holdout.py):

  in-sample ≈ held-out   → underfit (the model never learned the signal)
  in-sample ≫ held-out   → overfit  (the model memorized the training set)
  in-sample  >  held-out (moderate gap) → healthy — expected for a Random
                          Forest, whose bootstrap trees each see ~63% of rows.

    python src/insample_check.py   →   outputs/results/insample_check.json
"""

import sys, json, pathlib
import numpy as np
import joblib

sys.path.insert(0, "src")
import config

# These names must resolve in the module that unpickles the model: the saved
# estimator was pickled from train_ml, so its classes deserialize by reference.
from train_ml import (
    load_training_matrix, load_style_features,
    soft_metrics, ARCH_KEYS,
    FusionModel, SoftRegressor, MeanBaseline, LogRegBaseline,  # noqa: F401
)


def hard_metrics(y_true_hard, y_pred_hard, arch_keys=ARCH_KEYS):
    """Argmax accuracy + per-class precision / recall / F1 (in-sample)."""
    acc = float(np.mean(y_true_hard == y_pred_hard))
    per_class = {}
    for k, name in enumerate(arch_keys):
        tp = int(np.sum((y_true_hard == k) & (y_pred_hard == k)))
        fn = int(np.sum((y_true_hard == k) & (y_pred_hard != k)))
        fp = int(np.sum((y_true_hard != k) & (y_pred_hard == k)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {"n": int(tp + fn), "precision": round(prec, 4),
                           "recall": round(rec, 4), "f1": round(f1, 4)}
    return acc, per_class


def main():
    print("\n🔍  In-sample sanity check")
    print("=" * 50)

    bundle = joblib.load(config.ML_MODEL)
    model    = bundle["model"]
    fusion_w = bundle["fusion_metric_w"]
    print(f"  Model loaded: fusion metric_w={fusion_w:.2f} / style_w={1-fusion_w:.2f}")

    # Same assembly as train_ml.py, so the matrix is identical to what was fit.
    merged, X, y, w, hard_true, _ = load_training_matrix()
    S = load_style_features(merged)
    print(f"  Training users: {len(X)}")

    train_pred = model.predict(X, S)
    hard_pred  = train_pred.argmax(axis=1)

    kl, cos, brier, sacc = soft_metrics(y, train_pred)
    hard_acc, per_class   = hard_metrics(hard_true, hard_pred)

    per_class_sacc = {}
    for k, name in enumerate(ARCH_KEYS):
        mask = hard_true == k
        if mask.sum() > 0:
            _, _, _, s = soft_metrics(y[mask], train_pred[mask])
            per_class_sacc[name] = round(s, 4)

    print("\n  ── In-sample (training data) ──────────────────")
    print(f"  Soft-accuracy : {sacc:.4f}  ({sacc*100:.1f}%)")
    print(f"  Cosine sim    : {cos:.4f}")
    print(f"  KL divergence : {kl:.4f}")
    print(f"  Hard accuracy : {hard_acc:.4f}  ({hard_acc*100:.1f}%)")

    # Held-out reference (written earlier by compare_holdout.py, if present).
    print("\n  ── Blind held-out (reference) ─────────────────")
    holdout = {}
    try:
        with open(config.HOLDOUT_COMPARISON) as f:
            holdout = json.load(f)
        ho_sacc = holdout.get("soft_accuracy") or holdout.get("sacc")
        ho_cos  = holdout.get("cosine") or holdout.get("mean_cosine") or holdout.get("cos")
        if ho_sacc:
            print(f"  Soft-accuracy : {ho_sacc:.4f}  ({ho_sacc*100:.1f}%)")
        if ho_cos:
            print(f"  Cosine sim    : {ho_cos:.4f}")
    except Exception:
        print("  (holdout_comparison.json not found — run compare_holdout.py first)")

    print("\n  ── Per-class in-sample soft-accuracy ──────────")
    for name, s in per_class_sacc.items():
        print(f"  {name:<12} {s:.3f}  {'█' * int(s * 30)}")

    ho_sacc = holdout.get("soft_accuracy")
    out = {
        "n_train": int(len(X)),
        "fusion_metric_w": float(fusion_w),
        "insample": {
            "soft_accuracy": round(sacc, 4),
            "cosine": round(cos, 4),
            "kl_divergence": round(kl, 4),
            "brier_score": round(brier, 4),
            "hard_accuracy": round(hard_acc, 4),
            "per_class_f1": per_class,
            "per_class_soft_accuracy": per_class_sacc,
        },
        "holdout_reference": holdout if holdout else "not_found",
        "interpretation": {
            "insample_vs_holdout": (
                f"in-sample soft-acc {sacc:.3f} vs held-out {ho_sacc:.3f} "
                f"— gap = {sacc - ho_sacc:.3f}"
                if ho_sacc else "holdout data not available"
            )
        },
    }

    out_path = pathlib.Path(config.INSAMPLE_CHECK)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅  Saved → {out_path}")


if __name__ == "__main__":
    main()
