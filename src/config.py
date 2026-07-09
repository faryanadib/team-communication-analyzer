"""
config.py — single source of truth for every data / output path.

Paths are resolved relative to the PROJECT ROOT via __file__, so scripts work
no matter what the current working directory is (run `python src/train_ml.py`
from the repo root, or from anywhere). Importing this module also creates the
output directories if they are missing.

Layout
------
  data/raw/        original exports (WhatsApp .txt, Nankani .txt, Mbada .docx, Slack)
  data/clean/      parsed one-JSON-per-line message streams
  data/features/   13-metric matrices (v1 and v2)
  data/labels/     LLM soft labels + blind held-out gold
  outputs/models/  fitted model + embedding cache
  outputs/results/ predictions, CV reports, evaluation JSONs
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # repo root (src/ -> ..)

DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
FEATURES = DATA / "features"
LABELS = DATA / "labels"

OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
RESULTS = OUTPUTS / "results"
ANNOTATION = OUTPUTS / "annotation"   # human inter-rater validation study artifacts

for _d in (CLEAN, FEATURES, LABELS, MODELS, RESULTS, ANNOTATION):
    _d.mkdir(parents=True, exist_ok=True)

# ── raw ──────────────────────────────────────────────────────────────────────
NANKANI_RAW = str(RAW / "Nankani_2020.txt")
WHATSAPP_RAW = str(RAW / "whatsapp_groupchat.txt")
SLACK_RAW_DIR = str(RAW / "slack_data")
GARIMELLA_RAW = str(RAW / "Garimella_Kiran_15_6_2020_anonymized.csv.gz")

# ── clean message streams ────────────────────────────────────────────────────
SLACK_CLEAN = str(CLEAN / "slack_clean.json")
NANKANI_CLEAN = str(CLEAN / "nankani_clean.json")
MBADA_CLEAN = str(CLEAN / "mbada_clean.json")
GARIMELLA_CLEAN = str(CLEAN / "garimella_clean.json")
SYNTHETIC_CLEAN = str(CLEAN / "synthetic_clean.json")
WHATSAPP_CLEAN = str(CLEAN / "whatsapp_clean.json")
# NOTE: garimella is intentionally NOT in CLEAN_FILES — the dataset is broadcast/
# spam/propaganda, off-domain for team archetypes (see docs note). Kept available
# for the extractor only, not wired into the training pipeline.
# synthetic IS included: disclosed in-domain augmentation for the rare classes
# (see make_synthetic.py); train_ml can ablate it out via --no-synthetic.
CLEAN_FILES = {"slack": SLACK_CLEAN, "nankani": NANKANI_CLEAN,
               "mbada": MBADA_CLEAN, "synthetic": SYNTHETIC_CLEAN}

# ── feature matrices (13 metrics) ────────────────────────────────────────────
TRAIN_FEATURES = str(FEATURES / "training_results.json")        # v1
TRAIN_FEATURES_V2 = str(FEATURES / "training_results_v2.json")  # v2 (production)
WA_FEATURES = str(FEATURES / "wa_features.json")                # v1 team
WA_FEATURES_V2 = str(FEATURES / "wa_features_v2.json")          # v2 team (production)

# ── labels ───────────────────────────────────────────────────────────────────
LLM_LABELS = str(LABELS / "llm_labels.json")
WA_HOLDOUT = str(LABELS / "wa_holdout_llm.json")
# The weak-supervision labeler and its consumers read/write the same soft-label
# file; OPUS_LABELS is kept as an alias so the labeling scripts stay readable.
OPUS_LABELS = LLM_LABELS

# ── outputs: models ──────────────────────────────────────────────────────────
ML_MODEL = str(MODELS / "ml_model.joblib")
TEXT_CACHE = str(MODELS / "text_cache.npz")

# ── outputs: results ─────────────────────────────────────────────────────────
WA_RESULTS = str(RESULTS / "wa_results.json")
ML_RESULTS = str(RESULTS / "ml_results.json")
HOLDOUT_COMPARISON = str(RESULTS / "holdout_comparison.json")
INSAMPLE_CHECK = str(RESULTS / "insample_check.json")

# ── outputs: human inter-rater validation study ──────────────────────────────
ANNOTATION_SAMPLE = str(ANNOTATION / "sample_users.json")                 # blinded, shareable
ANNOTATION_ANSWER_KEY = str(ANNOTATION / "answer_key_DO_NOT_SHARE.json")  # model labels, kept hidden
ANNOTATION_ASSIGNMENT = str(ANNOTATION / "assignment.json")              # study design metadata
ANNOTATION_RETURNED = str(ANNOTATION / "returned")                       # filled CSVs come back here
ANNOTATION_AGREEMENT = str(RESULTS / "annotation_agreement.json")        # computed IAA report
