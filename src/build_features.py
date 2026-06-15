"""
build_features.py — build the base feature matrix for the real datasets.

Runs the shared `features.extract_features` (13 behaviour/network metrics) over
every available clean message stream, tags each row with its `dataset`, and
writes the combined matrix to `training_results.json`.

    data/clean/<dataset>_clean.json  →  data/features/training_results.json

This is the FIRST build step. Afterwards:
    • build_synthetic_features.py  appends the disclosed synthetic rows
    • build_v2_features.py         recomputes the corrected interaction metrics

Only datasets whose clean file exists are included — missing ones are skipped
with a warning, so you can train on whatever subset of data you have. The
synthetic augmentation is handled by its own builder and is skipped here.

    python src/build_features.py
"""
import os
import pandas as pd

import config
from features import (
    ALL_METRICS, ABSOLUTE_METRICS, RANK_METRICS,
    load_clean, extract_features, sanity_check_features,
)

# real datasets only — the synthetic augmentation has a dedicated builder
REAL_DATASETS = {k: v for k, v in config.CLEAN_FILES.items() if k != "synthetic"}


def main():
    print("\n🔎  Feature extraction — 13 behaviour/network metrics")
    print("=" * 64)
    print(f"  Absolute-scored ({len(ABSOLUTE_METRICS)}): {ABSOLUTE_METRICS}")
    print(f"  Rank-scored    ({len(RANK_METRICS)}): {RANK_METRICS}\n")

    frames = []
    for name, path in REAL_DATASETS.items():
        if not os.path.exists(path):
            print(f"  ⚠ skip '{name}' — clean file not found ({path})")
            continue
        df = load_clean(path)
        feats = extract_features(df, label=name)
        sanity_check_features(feats, label=name)
        feats["dataset"] = name
        frames.append(feats[["author", "dataset"] + ALL_METRICS])
        print(f"  ✓ {name}: {feats.shape[0]} users")

    if not frames:
        raise SystemExit(
            "\n  ✗ No clean datasets found. Parse your raw exports first, e.g.\n"
            "      python src/parse_slack.py / parse_nankani.py / parse_whatsapp.py")

    out = pd.concat(frames, ignore_index=True)
    out.to_json(config.TRAIN_FEATURES, orient="records", lines=True)
    print(f"\n  ✅ Base feature matrix {out.shape} → {config.TRAIN_FEATURES}")
    print("  Next:  python src/build_synthetic_features.py  (optional)")
    print("         python src/build_v2_features.py")


if __name__ == "__main__":
    main()
