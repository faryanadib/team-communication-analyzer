"""
analyze_agreement.py
════════════════════
Compute the inter-rater agreement report for the human validation study.

Reads:
  outputs/annotation/answer_key_DO_NOT_SHARE.json   (Opus soft labels)
  outputs/annotation/returned/*.csv                 (one CSV per human annotator,
                                                     exported from the HTML app)

Produces a printed report + outputs/results/annotation_agreement.json with:

  • SOFT metrics (full 5-way distribution, the way the project is judged):
      - soft-accuracy  Σ min(p,q)   (== the headline training metric)
      - cosine similarity
      - Jensen–Shannon divergence
  • HARD metrics (dominant archetype = argmax):
      - raw agreement %
      - Cohen's κ (pairwise), Fleiss' κ (all raters)
  • Reliability:
      - Krippendorff's α  (interval, per-archetype + averaged; nominal on argmax)
      - computed WITH and WITHOUT Opus, so we can see Opus does not lower it
  • The headline comparison:  human↔Opus  vs  human↔human
    If they are close, Opus annotates within human inter-rater reliability →
    the soft labels are trustworthy enough to weak-supervise the model.
  • Agreement broken down by Opus confidence level (high/medium/low).

Humans are NOT treated as ground truth — they are a second annotator; this is an
inter-rater reliability study, not an accuracy test against a gold standard.

Usage:
    python src/analyze_agreement.py
    python src/analyze_agreement.py --returned path/to/csvs   # custom folder
"""

import os
import csv
import glob
import json
import argparse
import itertools
from collections import defaultdict

import config

ARCHETYPE_KEYS = ["bee", "ant", "butterfly", "capybara", "leech"]


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════
def load_answer_key(path):
    """{(user_id, dataset): {probs, confidence}} from the Opus answer key."""
    key = {}
    for r in json.load(open(path, encoding="utf-8")):
        probs = {k: float(r[k]) for k in ARCHETYPE_KEYS}      # already sum to 1
        key[(r["user_id"], r["dataset"])] = {
            "probs": probs, "confidence": r.get("opus_confidence", "low")}
    return key


def load_annotator_csvs(folder):
    """{annotator: {(user_id, dataset): {probs, confidence}}}. Scores /100 → prob."""
    out = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for fp in files:
        with open(fp, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ann = (row.get("annotator") or os.path.basename(fp)).strip()
                raw = {k: float(row[k]) for k in ARCHETYPE_KEYS}
                tot = sum(raw.values()) or 1.0
                probs = {k: raw[k] / tot for k in ARCHETYPE_KEYS}
                out[ann][(row["user_id"], row["dataset"])] = {
                    "probs": probs, "confidence": (row.get("confidence") or "").strip()}
    return out, files


# ════════════════════════════════════════════════════════════════════════════
# Metric primitives
# ════════════════════════════════════════════════════════════════════════════
def vec(d):
    return [d[k] for k in ARCHETYPE_KEYS]


def soft_accuracy(p, q):
    return sum(min(a, b) for a, b in zip(vec(p), vec(q)))


def cosine(p, q):
    a, b = vec(p), vec(q)
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _kl(a, b):
    import math
    s = 0.0
    for x, y in zip(a, b):
        if x > 0 and y > 0:
            s += x * math.log2(x / y)
    return s


def jsd(p, q):
    """Jensen–Shannon divergence (base 2), in [0,1]."""
    a, b = vec(p), vec(q)
    m = [(x + y) / 2 for x, y in zip(a, b)]
    return 0.5 * _kl(a, m) + 0.5 * _kl(b, m)


def argmax_label(p):
    return max(ARCHETYPE_KEYS, key=lambda k: p[k])


def cohen_kappa(labels_a, labels_b):
    """Cohen's κ for two raters' nominal labels (aligned lists)."""
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = {k: labels_a.count(k) / n for k in ARCHETYPE_KEYS}
    cb = {k: labels_b.count(k) / n for k in ARCHETYPE_KEYS}
    pe = sum(ca[k] * cb[k] for k in ARCHETYPE_KEYS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(label_lists):
    """Fleiss' κ. label_lists: list (per unit) of lists of category labels from
    all raters who rated that unit. Units may have different numbers of raters."""
    cats = ARCHETYPE_KEYS
    rows = [lst for lst in label_lists if len(lst) >= 2]
    if not rows:
        return None
    # P_i per unit
    Pis, n_im_sum, totals = [], defaultdict(int), 0
    for lst in rows:
        n = len(lst)
        counts = {c: lst.count(c) for c in cats}
        Pi = (sum(v * v for v in counts.values()) - n) / (n * (n - 1)) if n > 1 else 0
        Pis.append(Pi)
        for c in cats:
            n_im_sum[c] += counts[c]
        totals += n
    Pbar = sum(Pis) / len(Pis)
    pj = {c: n_im_sum[c] / totals for c in cats}
    Pe = sum(v * v for v in pj.values())
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def krippendorff_alpha(units, metric="interval"):
    """Krippendorff's α. `units` = list of lists; each inner list holds the values
    one unit received from the raters present (>=2). For 'interval' values are
    floats; for 'nominal' they are hashable category labels."""
    if metric == "interval":
        delta = lambda c, k: (c - k) ** 2
    else:
        delta = lambda c, k: 0.0 if c == k else 1.0

    pairable = [u for u in units if len(u) >= 2]
    flat = [v for u in pairable for v in u]
    n = len(flat)
    if n < 2:
        return None

    # Observed disagreement
    Do_num = 0.0
    for u in pairable:
        m = len(u)
        inner = sum(delta(u[i], u[j]) for i in range(m) for j in range(m) if i != j)  # ordered pairs
        Do_num += inner / (m - 1)
    Do = Do_num / n

    # Expected disagreement over all ordered pairs in the flat value list
    De_num = sum(delta(a, b) for a in flat for b in flat) - sum(delta(v, v) for v in flat)
    De = De_num / (n * (n - 1))

    if De == 0:
        return 1.0
    return 1 - Do / De


# ════════════════════════════════════════════════════════════════════════════
# Pairwise + multi-rater aggregation
# ════════════════════════════════════════════════════════════════════════════
def pairwise(rater_a, rater_b, keys):
    """Aggregate soft + hard metrics for two raters over their common users."""
    common = [k for k in keys if k in rater_a and k in rater_b]
    if not common:
        return None
    sa = [soft_accuracy(rater_a[k]["probs"], rater_b[k]["probs"]) for k in common]
    cs = [cosine(rater_a[k]["probs"], rater_b[k]["probs"]) for k in common]
    js = [jsd(rater_a[k]["probs"], rater_b[k]["probs"]) for k in common]
    la = [argmax_label(rater_a[k]["probs"]) for k in common]
    lb = [argmax_label(rater_b[k]["probs"]) for k in common]
    raw = sum(1 for a, b in zip(la, lb) if a == b) / len(common)
    return {
        "n": len(common),
        "soft_accuracy": round(sum(sa) / len(sa), 4),
        "cosine": round(sum(cs) / len(cs), 4),
        "jsd": round(sum(js) / len(js), 4),
        "hard_agreement": round(raw, 4),
        "cohen_kappa": round(cohen_kappa(la, lb), 4) if len(common) else None,
    }


def mean_of(dlist, field):
    vals = [d[field] for d in dlist if d and d.get(field) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Compute human↔Opus inter-rater agreement.")
    ap.add_argument("--returned", default=config.ANNOTATION_RETURNED,
                    help="folder of annotator CSVs exported from the app")
    ap.add_argument("--answer-key", default=config.ANNOTATION_ANSWER_KEY)
    ap.add_argument("--out", default=config.ANNOTATION_AGREEMENT)
    args = ap.parse_args()

    opus = load_answer_key(args.answer_key)
    annotators, files = load_annotator_csvs(args.returned)
    if not annotators:
        print(f"No annotator CSVs found in {args.returned}/ — collect them first.")
        print("(Each teammate clicks ⬇️ دانلودِ CSV in the app and sends the file.)")
        return

    names = sorted(annotators)
    print("═" * 68)
    print("  HUMAN INTER-RATER VALIDATION OF THE LLM SOFT LABELS")
    print("═" * 68)
    print(f"  Annotators: {names}  (files: {[os.path.basename(f) for f in files]})")
    print(f"  Opus answer-key users: {len(opus)}")
    for a in names:
        print(f"    {a:<14} labeled {len(annotators[a])} users")

    # universe = users every needed rater covers (intersection for clean comparison)
    common_keys = set(opus)
    for a in names:
        common_keys &= set(annotators[a])
    common_keys = sorted(common_keys)
    print(f"  Users common to ALL raters: {len(common_keys)}")
    if not common_keys:
        print("  ⚠ no overlap yet — nothing to compare.")
        return

    report = {"annotators": names, "n_common": len(common_keys), "pairwise": {}}

    # ── 1. each human vs Opus ──────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  1) HUMAN  ↔  OPUS")
    print("─" * 68)
    h_vs_opus = []
    for a in names:
        m = pairwise(annotators[a], opus, common_keys)
        h_vs_opus.append(m)
        report["pairwise"][f"{a}_vs_opus"] = m
        print(f"  {a:<12} soft-acc {m['soft_accuracy']:.3f} | cos {m['cosine']:.3f} | "
              f"JSD {m['jsd']:.3f} | hard {m['hard_agreement']*100:4.0f}% | κ {m['cohen_kappa']}")

    # ── 2. human vs human (the reliability ceiling) ───────────────────────────
    print("\n" + "─" * 68)
    print("  2) HUMAN  ↔  HUMAN   (inter-annotator ceiling)")
    print("─" * 68)
    h_vs_h = []
    for a, b in itertools.combinations(names, 2):
        m = pairwise(annotators[a], annotators[b], common_keys)
        h_vs_h.append(m)
        report["pairwise"][f"{a}_vs_{b}"] = m
        print(f"  {a} ↔ {b:<8} soft-acc {m['soft_accuracy']:.3f} | cos {m['cosine']:.3f} | "
              f"JSD {m['jsd']:.3f} | hard {m['hard_agreement']*100:4.0f}% | κ {m['cohen_kappa']}")
    if not h_vs_h:
        print("  (only one annotator — cannot estimate the human ceiling; add a 2nd)")

    # ── 3. headline comparison ─────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  3) HEADLINE  —  is Opus within human reliability?")
    print("─" * 68)
    head = {
        "human_vs_opus": {f: mean_of(h_vs_opus, f) for f in
                          ["soft_accuracy", "cosine", "jsd", "hard_agreement", "cohen_kappa"]},
        "human_vs_human": {f: mean_of(h_vs_h, f) for f in
                           ["soft_accuracy", "cosine", "jsd", "hard_agreement", "cohen_kappa"]},
    }
    report["headline"] = head
    for f in ["soft_accuracy", "cosine", "hard_agreement", "cohen_kappa"]:
        ho, hh = head["human_vs_opus"][f], head["human_vs_human"][f]
        verdict = ""
        if ho is not None and hh is not None and hh != 0:
            ratio = ho / hh if hh else 0
            verdict = "✓ within human range" if ratio >= 0.9 else "~ slightly below human"
        print(f"  {f:<16} human↔Opus={ho}   human↔human={hh}   {verdict}")

    # ── 4. Krippendorff's α (with vs without Opus) ─────────────────────────────
    print("\n" + "─" * 68)
    print("  4) KRIPPENDORFF'S α  (reliability)")
    print("─" * 68)

    def alpha_block(include_opus):
        raters = list(names) + (["__opus__"] if include_opus else [])
        rater_data = {a: annotators[a] for a in names}
        if include_opus:
            rater_data["__opus__"] = opus
        # interval α per dimension
        per_dim = {}
        for ki, dim in enumerate(ARCHETYPE_KEYS):
            units = []
            for key in common_keys:
                vals = [rater_data[r][key]["probs"][dim] for r in raters if key in rater_data[r]]
                if len(vals) >= 2:
                    units.append(vals)
            per_dim[dim] = krippendorff_alpha(units, "interval")
        avg = round(sum(v for v in per_dim.values() if v is not None) /
                    max(1, len([v for v in per_dim.values() if v is not None])), 4)
        # nominal α on argmax
        units_nom = []
        for key in common_keys:
            labs = [argmax_label(rater_data[r][key]["probs"]) for r in raters if key in rater_data[r]]
            if len(labs) >= 2:
                units_nom.append(labs)
        nom = krippendorff_alpha(units_nom, "nominal")
        return {"interval_per_dim": {k: round(v, 4) if v is not None else None
                                     for k, v in per_dim.items()},
                "interval_avg": avg, "nominal_argmax": round(nom, 4) if nom is not None else None}

    a_all = alpha_block(True)
    a_hum = alpha_block(False)
    report["krippendorff"] = {"with_opus": a_all, "humans_only": a_hum}
    print(f"  interval α (soft, avg over archetypes):  with Opus {a_all['interval_avg']}"
          f"   humans-only {a_hum['interval_avg']}")
    print(f"  nominal  α (argmax archetype):           with Opus {a_all['nominal_argmax']}"
          f"   humans-only {a_hum['nominal_argmax']}")

    # ── 5. Fleiss' κ over all raters (hard) ───────────────────────────────────
    raters_all = list(names) + ["__opus__"]
    rdata = {**{a: annotators[a] for a in names}, "__opus__": opus}
    fleiss_units = []
    for key in common_keys:
        labs = [argmax_label(rdata[r][key]["probs"]) for r in raters_all if key in rdata[r]]
        if len(labs) >= 2:
            fleiss_units.append(labs)
    fk = fleiss_kappa(fleiss_units)
    report["fleiss_kappa_all_raters"] = round(fk, 4) if fk is not None else None
    print(f"  Fleiss' κ (all raters incl. Opus, hard): {report['fleiss_kappa_all_raters']}")

    # ── 6. agreement by Opus confidence ────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  5) HUMAN↔OPUS soft-accuracy  BY OPUS CONFIDENCE")
    print("─" * 68)
    by_conf = {}
    for level in ("high", "medium", "low"):
        keys_l = [k for k in common_keys if opus[k]["confidence"] == level]
        if not keys_l:
            continue
        vals = [soft_accuracy(annotators[a][k]["probs"], opus[k]["probs"])
                for a in names for k in keys_l]
        by_conf[level] = {"n_users": len(keys_l), "soft_accuracy": round(sum(vals) / len(vals), 4)}
        print(f"  {level:<8} n={len(keys_l):>2}  soft-acc {by_conf[level]['soft_accuracy']:.3f}")
    report["by_opus_confidence"] = by_conf

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Full report → {args.out}")


if __name__ == "__main__":
    main()
