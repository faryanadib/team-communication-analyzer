"""
validate_labels.py
══════════════════
Standalone QA report for the Opus soft labels (opus_labels.json).

Run after opus_labeler.py:
    python validate_labels.py

It prints:
  1. Coverage + dominant-archetype distribution (ASCII bars) + confidence mix
     + mean soft score per archetype + the list of low-confidence users.
  2. Agreement between Opus (argmax of soft scores) and the OLD rule-based labels
     in training_results.json, for users present in both — a convergent-validity
     check that the two independent labelers see similar patterns.
  3. A sample of 5 labeled users with full score vector + notes.

Reads no external services; safe to run anytime.
"""

import os
import json
from collections import Counter
import config

OPUS_FILE = config.OPUS_LABELS
RULE_FILE = config.TRAIN_FEATURES
ARCHETYPE_KEYS = ["bee", "ant", "butterfly", "capybara", "leech"]

# training_results.json stores the rule label as an emoji string → friendly key.
EMOJI_TO_KEY = {
    "🐝 Bee": "bee", "🐜 Ant": "ant", "🦋 Butterfly": "butterfly",
    "🦫 Capybara": "capybara", "🔴 Leech": "leech",
}


def load_jsonl(path):
    """Load a one-object-per-line JSON file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dominant(record):
    """Friendly key of the archetype with the highest soft score."""
    return max(ARCHETYPE_KEYS, key=lambda k: record[k])


def bar(frac, width=40):
    return "█" * int(round(frac * width))


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Opus label summary
# ════════════════════════════════════════════════════════════════════════════
def summarize_opus(records):
    n = len(records)
    print("═" * 64)
    print(f"  OPUS LABEL SUMMARY — {OPUS_FILE}  ({n} labeled users)")
    print("═" * 64)

    # Per-dataset counts.
    per_ds = Counter(r["dataset"] for r in records)
    print("  Labeled users per dataset:")
    for ds, c in sorted(per_ds.items()):
        print(f"    {ds:<10} {c}")

    # Dominant-archetype distribution with ASCII bars.
    dom = Counter(dominant(r) for r in records)
    print("\n  Dominant archetype distribution:")
    for k in ARCHETYPE_KEYS:
        c = dom.get(k, 0)
        print(f"    {k:<10} {c:>5} ({c / n * 100:5.1f}%) {bar(c / n)}")

    # Confidence levels.
    conf = Counter(r.get("confidence", "low") for r in records)
    print("\n  Confidence distribution:")
    for level in ("high", "medium", "low"):
        c = conf.get(level, 0)
        print(f"    {level:<8} {c:>5} ({c / n * 100:5.1f}%)")

    # Mean soft score per archetype (systematic over/under-scoring check).
    print("\n  Mean soft score per archetype (should each ≈ baseline 0.20):")
    for k in ARCHETYPE_KEYS:
        mean = sum(r[k] for r in records) / n
        print(f"    {k:<10} {mean:.3f} {bar(mean)}")

    # Low-confidence users (candidates for manual review).
    low = [r for r in records if r.get("confidence") == "low"]
    print(f"\n  Low-confidence users ({len(low)}):")
    if low:
        for r in low[:30]:
            print(f"    {r['dataset']:<8} {str(r['user_id'])[:20]:<20} "
                  f"→ {dominant(r)}  ({r.get('notes', '')[:60]})")
        if len(low) > 30:
            print(f"    … and {len(low) - 30} more")
    else:
        print("    (none)")


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Opus vs rule-based agreement
# ════════════════════════════════════════════════════════════════════════════
def agreement(records):
    print("\n" + "═" * 64)
    print("  CONVERGENT VALIDITY — Opus vs rule-based labels")
    print("═" * 64)
    if not os.path.exists(RULE_FILE):
        print(f"  {RULE_FILE} not found — skipping agreement check.")
        return

    rule = load_jsonl(RULE_FILE)
    # key on (author, dataset) to avoid cross-dataset name collisions.
    rule_map = {(r["author"], r.get("dataset")): EMOJI_TO_KEY.get(r.get("archetype"))
                for r in rule}

    matches = total = 0
    for r in records:
        key = (r["user_id"], r["dataset"])
        rule_label = rule_map.get(key)
        if rule_label is None:
            continue
        total += 1
        if dominant(r) == rule_label:
            matches += 1

    if total == 0:
        print("  No overlapping users between Opus labels and rule labels.")
        return

    pct = matches / total * 100
    print(f"  Opus vs Rule-based agreement: {matches}/{total} ({pct:.1f}%)")
    if pct < 50:
        print("  ⚠ LOW AGREEMENT — rule-based and Opus see very different patterns")
    elif pct > 80:
        print("  ✓ HIGH AGREEMENT — labels are consistent")
    else:
        print("  ~ MODERATE AGREEMENT — partial overlap (expected; labelers differ)")


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — sample of labeled users
# ════════════════════════════════════════════════════════════════════════════
def sample(records, k=5):
    print("\n" + "═" * 64)
    print(f"  SAMPLE — {min(k, len(records))} labeled users")
    print("═" * 64)
    for r in records[:k]:
        vec = " ".join(f"{a}={r[a]:.2f}" for a in ARCHETYPE_KEYS)
        print(f"  {r['dataset']:<8} {str(r['user_id'])[:20]:<20} [{r.get('confidence')}]")
        print(f"      {vec}")
        print(f"      notes: {r.get('notes', '')}")


def main():
    if not os.path.exists(OPUS_FILE):
        print(f"opus_labels.json not found — run `python opus_labeler.py` first.")
        return
    records = load_jsonl(OPUS_FILE)
    if not records:
        print(f"{OPUS_FILE} is empty — nothing to validate.")
        return
    summarize_opus(records)
    agreement(records)
    sample(records)


if __name__ == "__main__":
    main()
