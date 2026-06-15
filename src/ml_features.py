"""
ml_features.py
══════════════
ML-backed detectors for the three metrics that a plain word list / timing rule
cannot capture precisely:

    • emotion_density   — is a message emotional?      (6-emotion classifier)
    • harmony_score     — is a message supportive?     (sentiment / positivity)
    • initiation_rate   — does a message open a NEW topic? (timing + semantics)

Each detector AUTO-SELECTS the best backend that is actually installed, and
falls back gracefully so the pipeline always runs:

    emotion : transformers emotion model
              → NRCLex (NRC Emotion Lexicon)
              → built-in word list
    harmony : transformers sentiment model
              → VADER (Hutto & Gilbert 2014, tuned rule-based, offline)
              → built-in word list
    topic   : sentence-transformers embeddings (semantic topic-shift)
              → (unavailable → timing-only initiation)

Scientific sources:
  • Emotion model  : Hartmann (2022) emotion-english-distilroberta-base,
                     trained on 6 Ekman emotions; lexicon proxy = Mohammad &
                     Turney (2013) NRC Emotion Lexicon.
  • Sentiment model: Barbieri et al. (2020) TweetEval / cardiffnlp RoBERTa.
  • VADER          : Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious
                     Rule-based Model for Sentiment Analysis of Social Media
                     Text. ICWSM 2014.
  • Sentence embeddings: Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019.

The transformer backends need a one-time model download (internet) and are
CPU-slow on large corpora, so callers should sample per user on big datasets
(see SAMPLE_WHEN_NEURAL). VADER / NRCLex / word lists are offline and fast.
"""

import numpy as np

# ── Tunables ─────────────────────────────────────────────────────────────────
VADER_POS_THRESHOLD    = 0.30   # compound ≥ +0.30 → supportive/positive message
NEURAL_EMOTION_MINPROB = 0.50   # top emotion must beat this to count as "emotional"
TOPIC_SHIFT_SIM        = 0.30   # cosine sim below this vs previous msg → new topic

# Groups with ≤ this many users are scored in full (no per-user sampling).
_SMALL_GROUP_THRESHOLD = 30
# For large datasets, cap messages per user to keep RAM and time tractable.
_LARGE_DATASET_SAMPLE  = 20

# Legacy constant kept so external code that reads mlf.SAMPLE_WHEN_NEURAL still works.
SAMPLE_WHEN_NEURAL = _LARGE_DATASET_SAMPLE


def get_sample_size(n_users: int):
    """Return per-user message cap for neural scoring, or None for small groups."""
    return None if n_users <= _SMALL_GROUP_THRESHOLD else _LARGE_DATASET_SAMPLE


def _adaptive_batch(n_texts: int) -> int:
    """Smaller batch for larger inputs to avoid OOM on CPU."""
    if n_texts <= 300:
        return 16
    if n_texts <= 3000:
        return 8
    return 4

# ── Self-contained fallback lexicons (no import from features.py → no cycle) ──
HARMONY_WORDS = [
    "great", "well done", "good job", "thanks", "thank you", "appreciate",
    "agree", "exactly", "love", "perfect", "awesome", "nice", "good point",
    "helpful", "support", "welcome", "brilliant", "excellent", "congrats",
    "anytime", "no problem", "relax",
]
EMOTION_WORDS = [
    "happy", "joy", "love", "great", "wonderful", "excited", "glad", "enjoy",
    "pleasure", "delight", "fantastic", "amazing", "excellent", "brilliant",
    "wow", "surprised", "unexpected", "incredible", "unbelievable",
    "trust", "reliable", "honest", "support", "believe", "confident",
    "worried", "afraid", "nervous", "anxious", "concerned", "hope", "expect",
    "angry", "frustrated", "annoyed", "upset", "hate", "terrible", "awful",
    "sad", "sorry", "unfortunate", "disappointed", "miss", "regret",
]

# ── Backend detection (cheap import checks only; models load lazily) ──────────
def _has(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None

def _nrclex_works():
    """NRCLex needs nltk corpora; probe it so we don't silently score 0 offline."""
    if not _has("nrclex"):
        return False
    try:
        from nrclex import NRCLex
        aff = NRCLex("happy excited").affect_frequencies
        return isinstance(aff, dict) and len(aff) > 0
    except Exception:
        return False

_HAS_TRANSFORMERS = _has("transformers") and _has("torch")
_HAS_VADER        = _has("vaderSentiment")
_HAS_NRCLEX       = _nrclex_works()
_HAS_SBERT        = _has("sentence_transformers")

# Lazy singletons.
_emo_pipe = _sent_pipe = _vader = _sbert = None


def _emotion_pipe():
    global _emo_pipe
    if _emo_pipe is None:
        from transformers import pipeline
        _emo_pipe = pipeline("text-classification",
                             model="j-hartmann/emotion-english-distilroberta-base",
                             top_k=1, truncation=True)
    return _emo_pipe


def _sentiment_pipe():
    global _sent_pipe
    if _sent_pipe is None:
        from transformers import pipeline
        _sent_pipe = pipeline("sentiment-analysis",
                              model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                              truncation=True)
    return _sent_pipe


def _vader_analyzer():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _sbert_model():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert


# ── Backend choice (resolved once, logged by backend_report) ─────────────────
EMOTION_BACKEND = ("transformers" if _HAS_TRANSFORMERS else
                   "nrclex" if _HAS_NRCLEX else "wordlist")
HARMONY_BACKEND = ("transformers" if _HAS_TRANSFORMERS else
                   "vader" if _HAS_VADER else "wordlist")
TOPIC_BACKEND   = ("sentence-transformers" if _HAS_SBERT else "timing-only")
USES_NEURAL     = _HAS_TRANSFORMERS or _HAS_SBERT


def backend_report():
    return (f"ML backends → emotion_density: {EMOTION_BACKEND} · "
            f"harmony_score: {HARMONY_BACKEND} · "
            f"initiation topic-shift: {TOPIC_BACKEND}")


# ── Per-message flag computation, with de-duplication caching ────────────────
def _flags_with_cache(texts, scorer):
    """Apply `scorer(list_of_unique_texts)->list_of_0/1` with dedup caching."""
    uniq = list(dict.fromkeys(str(t) for t in texts))
    res = scorer(uniq)
    lut = dict(zip(uniq, res))
    return np.array([lut[str(t)] for t in texts], dtype=int)


def emotion_flags(texts):
    """1 if a message expresses an emotion, else 0 (per message)."""
    if EMOTION_BACKEND == "transformers":
        def scorer(uniq):
            bs = _adaptive_batch(len(uniq))
            out = _emotion_pipe()(uniq, batch_size=bs)
            flags = []
            for o in out:
                top = o[0] if isinstance(o, list) else o
                flags.append(int(top["label"].lower() != "neutral"
                                 and top["score"] >= NEURAL_EMOTION_MINPROB))
            return flags
        return _flags_with_cache(texts, scorer)

    if EMOTION_BACKEND == "nrclex":
        from nrclex import NRCLex
        def scorer(uniq):
            flags = []
            for t in uniq:
                try:
                    aff = NRCLex(t).affect_frequencies
                    emo = sum(v for k, v in aff.items()
                              if k not in ("positive", "negative"))
                    flags.append(int(emo > 0))
                except Exception:
                    flags.append(0)
            return flags
        return _flags_with_cache(texts, scorer)

    # wordlist fallback (whole-token match)
    def scorer(uniq):
        return [int(any(w in t.lower().split() for w in EMOTION_WORDS)) for t in uniq]
    return _flags_with_cache(texts, scorer)


def harmony_flags(texts):
    """1 if a message is supportive / positive, else 0 (per message)."""
    if HARMONY_BACKEND == "transformers":
        def scorer(uniq):
            bs = _adaptive_batch(len(uniq))
            out = _sentiment_pipe()(uniq, batch_size=bs)
            return [int(o["label"].lower().startswith("pos")) for o in out]
        return _flags_with_cache(texts, scorer)

    if HARMONY_BACKEND == "vader":
        v = _vader_analyzer()
        def scorer(uniq):
            return [int(v.polarity_scores(t)["compound"] >= VADER_POS_THRESHOLD)
                    for t in uniq]
        return _flags_with_cache(texts, scorer)

    # wordlist fallback (substring match — least precise)
    def scorer(uniq):
        return [int(any(w in t.lower() for w in HARMONY_WORDS)) for t in uniq]
    return _flags_with_cache(texts, scorer)


def topic_shift_flags(texts):
    """
    1 if a message starts a semantically NEW topic vs the previous message,
    else 0. Returns None if no embedding backend is available (→ caller uses
    timing only). Operates on the time-ordered message stream.
    """
    if not _HAS_SBERT or len(texts) < 2:
        return None
    emb = _sbert_model().encode([str(t) for t in texts], batch_size=64,
                                normalize_embeddings=True, show_progress_bar=False)
    sims = (emb[1:] * emb[:-1]).sum(axis=1)        # cosine of consecutive msgs
    flags = np.zeros(len(texts), dtype=int)
    flags[1:] = (sims < TOPIC_SHIFT_SIM).astype(int)
    return flags
