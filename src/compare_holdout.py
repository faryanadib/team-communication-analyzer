"""
compare_holdout.py — Blind held-out validation of the trained ML model.

Compares:
  (A) LLM gold labels  — read from raw text only, blind to the 13 metrics
                          (wa_holdout_llm.json)
  (B) ML model output   — RF trained on Slack+Nankani LLM labels, then run on
                          the team's 13 metrics (wa_results.json)

This is the ONLY honest test of the model: the model never saw these 5 people
during training, and LLM never saw the model's features when labeling them.

Reports per-person agreement + soft-distribution distance (KL / cosine), and a
confusion summary. Nothing here is fed back into training.
"""
import json
import math
import config

ARCH = ["bee", "ant", "butterfly", "capybara", "leech"]
EMOJI = {"bee": "🐝", "ant": "🐜", "butterfly": "🦋", "capybara": "🦫", "leech": "🔴"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def kl(p, q, eps=1e-9):
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def cosine(p, q):
    dot = sum(a * b for a, b in zip(p, q))
    np_ = math.sqrt(sum(a * a for a in p))
    nq = math.sqrt(sum(b * b for b in q))
    return dot / (np_ * nq) if np_ and nq else 0.0


def soft_accuracy(p, q):
    """Probability-mass overlap Σ min(pᵢ,qᵢ) ∈ [0,1] — a SOFT 'accuracy' that
    credits partial agreement on the whole distribution instead of only the
    argmax. 1.0 = identical distributions; equals 1 − ½·L1 distance."""
    return sum(min(a, b) for a, b in zip(p, q))


def main():
    gold = {d["author"]: d for d in json.load(open(config.WA_HOLDOUT))["labels"]}
    pred = {r["author"]: r for r in load_jsonl(config.WA_RESULTS)}

    order = sorted(gold.keys())

    print("=" * 74)
    print("BLIND HELD-OUT VALIDATION  —  LLM gold (text)  vs  ML model (metrics)")
    print("=" * 74)

    hits = 0
    kls, coss, saccs = [], [], []
    confusion = []  # (gold, pred)

    for name in order:
        g = gold[name]
        p = pred[name]
        gsoft = [g["soft"][a] for a in ARCH]
        psoft = [p[f"{a}_pct"] / 100.0 for a in ARCH]

        g_top = g["archetype"]
        p_top = max(ARCH, key=lambda a: p[f"{a}_pct"])
        agree = g_top == p_top
        hits += agree
        confusion.append((g_top, p_top))

        k = kl(gsoft, psoft)
        c = cosine(gsoft, psoft)
        sa = soft_accuracy(gsoft, psoft)
        kls.append(k)
        coss.append(c)
        saccs.append(sa)

        mark = "✅ MATCH" if agree else "❌ MISS "
        print(f"\n{name:8s} ({g['n_messages']:3d} msgs)   {mark}")
        print(f"   LLM  (text)   → {EMOJI[g_top]} {g_top:9s} [{g['confidence']}]")
        print(f"   Model (metric) → {EMOJI[p_top]} {p_top:9s} "
              f"[conf {p[p_top+'_pct']:.0f}%]")
        # side-by-side soft bars
        print(f"   {'arch':10s} {'LLM':>6s}  {'Model':>6s}")
        for a in ARCH:
            print(f"   {a:10s} {g['soft'][a]*100:5.0f}%  {p[a+'_pct']:5.1f}%")
        print(f"   KL(gold‖model)={k:.3f}   cosine={c:.3f}   soft-acc={sa*100:.0f}%")

    n = len(order)
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  Top-1 (hard) agreement : {hits}/{n}  ({100*hits/n:.0f}%)")
    print(f"  SOFT accuracy (mass overlap w/ LLM): {100*sum(saccs)/n:.1f}%   ← headline")
    print(f"  Mean cosine            : {sum(coss)/n:.3f}")
    print(f"  Mean KL  (gold‖model)  : {sum(kls)/n:.3f}")

    print("\n  Confusion (gold → model):")
    for gt, pt in confusion:
        flag = "" if gt == pt else "   <-- disagreement"
        print(f"    {EMOJI[gt]} {gt:9s} → {EMOJI[pt]} {pt:9s}{flag}")

    # where does the model systematically drift?
    model_leech = sum(1 for _, pt in confusion if pt == "leech")
    gold_leech = sum(1 for gt, _ in confusion if gt == "leech")
    print(f"\n  Leech predicted by model: {model_leech}/{n}   "
          f"|  Leech in LLM gold: {gold_leech}/{n}")
    if model_leech > gold_leech:
        print("  ⚠  Model over-predicts Leech on this 5-person group — consistent")
        print("     with the in/out-ratio denominator blow-up + domain shift from")
        print("     large training communities to a tiny team.")

    out = {
        "top1_agreement": hits / n,
        "soft_accuracy": sum(saccs) / n,
        "mean_kl": sum(kls) / n,
        "mean_cosine": sum(coss) / n,
        "per_person": [
            {"author": name,
             "llm": gold[name]["archetype"],
             "model": max(ARCH, key=lambda a: pred[name][f"{a}_pct"]),
             "match": gold[name]["archetype"] == max(ARCH, key=lambda a: pred[name][f"{a}_pct"]),
             "kl": kls[i], "cosine": coss[i], "soft_accuracy": saccs[i]}
            for i, name in enumerate(order)
        ],
    }
    json.dump(out, open(config.HOLDOUT_COMPARISON, "w"), indent=2)
    print("\n  → wrote holdout_comparison.json")


if __name__ == "__main__":
    main()
