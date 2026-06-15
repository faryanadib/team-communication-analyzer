"""
build_v2_features.py — produce training_results_v2.json + wa_features_v2.json.

Strategy: keep v1's 8 content metrics byte-for-byte and overwrite ONLY the 5
interaction columns with features_v2's corrected computation (per dataset, from
the raw *_clean.json message streams). This isolates the A/B comparison to the
five fixes. No ML backends are re-run (the 5 metrics are graph/timing/regex).
"""
import json
import pandas as pd
import features
import features_v2 as fv2
import config

CLEAN = config.CLEAN_FILES


def main():
    v1 = pd.read_json(config.TRAIN_FEATURES, lines=True)
    parts = []
    for ds, sub in v1.groupby("dataset"):
        if ds not in CLEAN:
            parts.append(sub); continue
        print(f"  {ds}: recomputing interaction metrics …", flush=True)
        df = features.load_clean(CLEAN[ds])
        merged = fv2.merge_v2(sub.reset_index(drop=True), df, verbose=True)
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)[list(v1.columns)]
    out.to_json(config.TRAIN_FEATURES_V2, orient="records", lines=True)
    print(f"  → {config.TRAIN_FEATURES_V2} ({len(out)} rows)")

    # team
    wa = pd.read_json(config.WA_FEATURES, lines=True)
    dfw = features.load_clean(config.WHATSAPP_CLEAN)
    wa_v2 = fv2.merge_v2(wa, dfw, verbose=True)
    wa_v2.to_json(config.WA_FEATURES_V2, orient="records", lines=True)
    print(f"  → {config.WA_FEATURES_V2} ({len(wa_v2)} rows)")


if __name__ == "__main__":
    main()
