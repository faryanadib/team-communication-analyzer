"""
build_synthetic_features.py — fold the synthetic augmentation into the v1 matrix.

Runs the SAME features.extract_features used for every real dataset on the
synthetic clean stream, tags the rows dataset="synthetic", and appends them to
training_results.json (idempotently — any prior synthetic rows are dropped
first). Run build_v2_features.py afterwards to refresh the v2 matrix.

The `archetype` hard column is set to the argmax of the LLM soft label
(label_synthetic.py); it is only used for reporting, never by the trained
model (which distils the soft labels in llm_labels.json).
"""
import json
import pandas as pd
import features
import config
from label_synthetic import SYNTH_LABELS   # single source of truth for the labels

ARCH = ["bee", "ant", "butterfly", "capybara", "leech"]


def main():
    df = features.load_clean(config.SYNTHETIC_CLEAN)
    feats = features.extract_features(df, label="Synthetic augmentation")
    feats["dataset"] = "synthetic"
    argmax = {u: max(ARCH, key=lambda a: lab[a]) for u, lab in SYNTH_LABELS.items()}
    feats["archetype"] = feats["author"].map(argmax)

    v1 = pd.read_json(config.TRAIN_FEATURES, lines=True)
    v1 = v1[v1["dataset"] != "synthetic"]                 # idempotent
    out = pd.concat([v1, feats[list(v1.columns)]], ignore_index=True)
    out.to_json(config.TRAIN_FEATURES, orient="records", lines=True)
    print(f"✅ training_results.json now {len(out)} rows "
          f"(+{len(feats)} synthetic)")


if __name__ == "__main__":
    main()
