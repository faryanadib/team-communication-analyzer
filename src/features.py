"""
features.py
═══════════
Shared feature-extraction module for the Team Communication Analyzer (SocialCompass).

Extracts the **13 active metrics** used by the COIN archetype classifier
(Bee, Ant, Butterfly, Capybara, Leech). These are the columns of the
reference behaviour table:

    1.  in_out_ratio            (absolute score)
    2.  avg_msg_length          (absolute score, WORDS per message)
    3.  replies_sent            (absolute score, RATE 0-1)
    4.  question_ratio          (absolute score)
    5.  avg_mentions            (absolute score)
    6.  task_focus_score        (absolute score, RATE 0-1)
    7.  harmony_score           (absolute score, RATE 0-1)   ← renamed capybara_score
    8.  emoji_ratio             (absolute score, RATE 0-1)
    9.  emotion_density         (absolute score, RATE 0-1)
    10. initiation_rate         (absolute score, RATE 0-1)
    11. MATTR                   (RANK-normalized score)
    12. betweenness_centrality  (RANK-normalized score)
    13. response_consistency    (RANK-normalized score)     ← NEW (inverse burstiness)

This module ONLY extracts raw numeric values; the supervised model
(`train_ml.py`) consumes them. Keeping extraction in one shared module means
every consumer computes the metrics identically and they can never drift out
of sync.

Scientific sources are cited inline against every metric.
"""

import re
import numpy as np
import pandas as pd
import networkx as nx

import ml_features as mlf   # ML-backed emotion / harmony / topic-shift detectors

# ── Metric groupings (content vs interaction metric groups) ──────────
# 10 metrics scored with ABSOLUTE thresholds (absolute bands)
ABSOLUTE_METRICS = [
    "in_out_ratio", "avg_msg_length", "replies_sent", "question_ratio",
    "avg_mentions", "task_focus_score", "harmony_score", "emoji_ratio",
    "emotion_density", "initiation_rate",
]
# 3 metrics scored with RANK-NORMALIZATION → within-group percentile → 0-4
# On a tiny 5-person team absolute thresholds are unfair
# for these network/linguistic measures, so we rank within the group instead.
RANK_METRICS = [
    "MATTR", "betweenness_centrality", "response_consistency",
]
ALL_METRICS = ABSOLUTE_METRICS + RANK_METRICS  # 13 active metrics

REPLY_WINDOW_MINS = 10   # a message ≤10 min after a different author = a reply
INIT_GAP_MINS     = 30   # a message >30 min after the previous = conversation start


# ════════════════════════════════════════════════════════════════════════════
# Lexicons (rule-based detection only — no semantic understanding)
# ════════════════════════════════════════════════════════════════════════════
TASK_WORDS = [
    "done", "finished", "completed", "sent", "attached", "will do", "deadline",
    "file", "link", "submitted", "ready", "push", "commit", "fixed", "update",
    "works", "working", "solved", "merged", "deployed", "closed", "pull",
    "kiya", "ho gaya", "kar diya",  # Hindi task words present in Nankani CODERS data
]

# harmony_score (supportive language) and emotion_density (emotional language)
# are now detected by ml_features.py (transformers → VADER/NRCLex → word list),
# which is far more precise than a raw substring word list. The fallback lexicons
# live there as a single source of truth.

QUESTION_WORDS = ["why", "what", "how", "when", "where", "who", "which", "?"]

# Unicode emoji ranges (rule-based emoji detection).
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001F9FF"   # symbols & pictographs
    "\U00002600-\U000027BF"    # misc symbols / dingbats
    "\U0001FA00-\U0001FA9F"    # extended-A
    "\U0001F000-\U0001F0FF"    # mahjong/dominoes/cards
    "]+",
    flags=re.UNICODE,
)


# ════════════════════════════════════════════════════════════════════════════
# Loading & reply detection
# ════════════════════════════════════════════════════════════════════════════
def load_clean(json_path):
    """Load a *_clean.json (one JSON object per line) produced by a parser."""
    df = pd.read_json(json_path, lines=True)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["body"] = df["body"].fillna("").astype(str)
    return df


def _add_reply_structure(df):
    """Annotate reply / initiation / got-reply flags on the message stream."""
    df = df.copy()
    df["prev_author"] = df["author"].shift(1)
    df["prev_time"]   = df["datetime"].shift(1)
    df["gap_mins"]    = (df["datetime"] - df["prev_time"]).dt.total_seconds() / 60.0

    # A reply = within REPLY_WINDOW_MINS of a *different* author's message.
    df["is_reply"] = (
        (df["gap_mins"] <= REPLY_WINDOW_MINS) & (df["prev_author"] != df["author"])
    ).astype(int)

    # An initiation = a message that opens a new conversation (long silence before).
    df["is_initiator"] = (df["gap_mins"] > INIT_GAP_MINS).astype(int)

    # "got_reply" = this message was answered by someone else within the window
    # → used for the IN side of in_out_ratio.
    df["next_author"] = df["author"].shift(-1)
    df["next_gap"]    = df["gap_mins"].shift(-1).fillna(9_999.0)
    df["got_reply"]   = (
        (df["next_gap"] <= REPLY_WINDOW_MINS) & (df["next_author"] != df["author"])
    ).astype(int)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Individual metric helpers
# ════════════════════════════════════════════════════════════════════════════
def compute_mattr(texts, window=20):
    """
    MATTR — Moving-Average Type-Token Ratio (vocabulary diversity).
    A fixed 20-word window slides across the member's messages; MATTR is the
    average share of unique words per window, keeping it independent of how much
    the member wrote.
    Source: Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian Knot:
            The Moving-Average Type-Token Ratio (MATTR). J. Quantitative
            Linguistics 17(2):94-100.
    Limitation: if a member wrote fewer words than the window, the value
    is unstable → we fall back to a whole-text type-token ratio and the caller
    flags low-token users.
    """
    tokens = " ".join(str(t) for t in texts).lower().split()
    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0
    ttrs = [len(set(tokens[i:i + window])) / window
            for i in range(len(tokens) - window + 1)]
    return round(float(np.mean(ttrs)), 4)


def compute_burstiness(gaps):
    """
    Burstiness B of a member's reply-latency sequence, with the finite-size
    correction for short sequences.
    B ∈ [-1, 1]: B→1 very bursty (erratic), B→-1 very regular, B≈0 Poisson.

    Finite-size-corrected form for n inter-event intervals:
        B(r, n) = (√(n+1)·r − √(n−1)) / ((√(n+1) − 2)·r + √(n−1)),  r = σ/μ
    Sources: Goh, K.-I., & Barabási, A.-L. (2008). Burstiness and Memory in
             Complex Systems. EPL 81(4):48002.
             Kim, E.-K., & Jo, H.-H. (2016). Measuring Burstiness for Finite
             Event Sequences. Phys. Rev. E 94(3):032311.
    """
    gaps = np.asarray([g for g in gaps if np.isfinite(g) and g >= 0], dtype=float)
    n = len(gaps)
    if n < 2:
        return np.nan            # undefined — caller treats as missing/flag
    mu, sigma = gaps.mean(), gaps.std()
    if mu == 0:
        return np.nan
    r = sigma / mu
    num = np.sqrt(n + 1) * r - np.sqrt(n - 1)
    den = (np.sqrt(n + 1) - 2) * r + np.sqrt(n - 1)
    if den == 0:
        return np.nan
    return float(num / den)


def _ml_fraction(df, flag_fn):
    """
    Per-author mean of a per-message 0/1 flag from ml_features.
    When a (slow) neural backend is active, score only the first
    SAMPLE_WHEN_NEURAL messages per user — the per-user fraction is estimated
    from that sample. Offline backends (VADER / word list) score everything.
    """
    _sample_n = mlf.get_sample_size(df["author"].nunique())
    work = (df.groupby("author", group_keys=False).head(_sample_n)
            if (mlf.USES_NEURAL and _sample_n is not None)
            else df)
    flags = flag_fn(work["body"].tolist())
    return pd.Series(flags, index=work.index).groupby(work["author"]).mean()


def compute_betweenness(df):
    """
    Betweenness centrality on the reply graph — the bridging / connector role.
    Members are linked when they reply to each other; betweenness measures how
    often a member sits on the shortest paths between other pairs (networkx,
    normalized to 0-1).
    Sources: Freeman, L. C. (1977). A Set of Measures of Centrality Based on
             Betweenness. Sociometry 40(1):35-41.
             Brandes, U. (2001). A Faster Algorithm for Betweenness Centrality.
             J. Math. Sociology 25(2):163-177.
             Gloor, P. A. (2017). Swarm Leadership and the Collective Mind.
    Limitation: on a 5-person network this is unstable and often 0 for
    most members — stated in the paper and flagged by the sanity checks.
    """
    G = nx.DiGraph()
    G.add_nodes_from(df["author"].unique())
    for i in range(1, len(df)):
        a, prev = df.iloc[i]["author"], df.iloc[i]["prev_author"]
        gap     = df.iloc[i]["gap_mins"]
        if a != prev and pd.notna(prev) and gap <= REPLY_WINDOW_MINS:
            if G.has_edge(a, prev):
                G[a][prev]["weight"] += 1
            else:
                G.add_edge(a, prev, weight=1)
    if G.number_of_edges() == 0:
        return {a: 0.0 for a in df["author"].unique()}
    return nx.betweenness_centrality(G, normalized=True)


# ════════════════════════════════════════════════════════════════════════════
# Main extraction
# ════════════════════════════════════════════════════════════════════════════
def extract_features(df, label="dataset", verbose=True):
    """
    Extract the 13 active metrics, one row per author.
    `df` must contain columns: author, datetime (ms epoch or datetime), body.
    Returns a DataFrame indexed 0..n with an `author` column + 13 metric columns.
    """
    if "datetime" in df.columns and not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["body"] = df["body"].fillna("").astype(str)
    df = _add_reply_structure(df)

    if verbose:
        print(f"  [features] {label}: {len(df):,} messages · {df['author'].nunique():,} users")
        print(f"  [features] {mlf.backend_report()}")

    g = df.groupby("author")
    body_lower = df["body"].str.lower()

    # ── 2. avg_msg_length — mean WORDS per message (mean words). ──
    df["word_count"] = df["body"].str.split().apply(len)
    avg_msg_length = g["word_count"].mean()

    # ── 3. replies_sent — RATE: proportion of a user's messages that are replies
    #      (a proportion, not a raw count). ────────────
    replies_sent = g["is_reply"].mean()

    # ── 4. question_ratio — fraction of messages that ask a question. ────────
    df["is_question"] = body_lower.apply(
        lambda t: int(("?" in t) or any(w in t.split() for w in QUESTION_WORDS))
    )
    question_ratio = g["is_question"].mean()

    # ── 5. avg_mentions — mean @-mentions per message. ──────────────────────
    df["mention_count"] = df["body"].str.count(r"@\w+")
    avg_mentions = g["mention_count"].mean()

    # ── 6. task_focus_score — fraction of messages with ≥1 task word. ───────
    df["has_task"] = body_lower.apply(lambda t: int(any(w in t for w in TASK_WORDS)))
    task_focus_score = g["has_task"].mean()

    # ── 7. harmony_score — fraction of SUPPORTIVE messages (ML: sentiment). ─
    #       Detected by ml_features (transformers sentiment → VADER → words).
    harmony_score = _ml_fraction(df, mlf.harmony_flags)

    # ── 8. emoji_ratio — fraction of messages containing ≥1 emoji. ─
    df["has_emoji"] = df["body"].apply(lambda t: int(bool(EMOJI_PATTERN.search(t))))
    emoji_ratio = g["has_emoji"].mean()

    # ── 9. emotion_density — fraction of EMOTIONAL messages (ML: emotion clf). ─
    #       Detected by ml_features (transformers emotion → NRCLex → words).
    emotion_density = _ml_fraction(df, mlf.emotion_flags)

    # ── 10. initiation_rate — fraction of a user's messages that OPEN a new
    #       conversation. Backbone is timing (>30 min gap); if an embedding
    #       backend is available, a semantic topic-shift also counts as an
    #       initiation (Reimers & Gurevych 2019). ──────────────────────────
    is_init = df["is_initiator"].to_numpy()
    shift = mlf.topic_shift_flags(df["body"].tolist())   # None if no SBERT
    if shift is not None:
        is_init = ((is_init == 1) | (shift == 1)).astype(int)
    initiation_rate = pd.Series(is_init, index=df.index).groupby(df["author"]).mean()

    # ── 1. in_out_ratio — received / sent (N-1 dropped for cross-dataset
    #      comparability). "received" = this user's
    #      messages that drew a reply; "sent" = reply messages this user sent. ─
    replies_received_cnt = g["got_reply"].sum()
    replies_sent_cnt     = g["is_reply"].sum()

    def _in_out(author):
        rcv = float(replies_received_cnt.get(author, 0))
        snt = float(replies_sent_cnt.get(author, 0))
        if snt == 0:                       # pure receiver who never replies
            return min(rcv, 4.0) if rcv > 0 else 0.0
        return round(rcv / snt, 3)
    in_out_ratio = pd.Series({a: _in_out(a) for a in df["author"].unique()})

    # ── 11. MATTR (rank-normalized later). ──────────────────────────────────
    if verbose:
        print("  [features] computing MATTR …", flush=True)
    mattr = g["body"].apply(lambda x: compute_mattr(x.tolist()))

    # ── 12. betweenness_centrality (rank-normalized later). ─────────────────
    betweenness = pd.Series(compute_betweenness(df))

    # ── 13. response_consistency = inverse burstiness of reply latencies.
    #       High = steady rhythm, low = erratic/bursty (rank-normalized later). ─
    reply_gaps = df[df["is_reply"] == 1].groupby("author")["gap_mins"]
    burst = reply_gaps.apply(lambda s: compute_burstiness(s.tolist()))
    # consistency = -burstiness; undefined (too few replies) → neutral 0.0, flagged.
    response_consistency = (-burst).reindex(df["author"].unique())

    feats = pd.DataFrame({
        "in_out_ratio":           in_out_ratio,
        "avg_msg_length":         avg_msg_length,
        "replies_sent":           replies_sent,
        "question_ratio":         question_ratio,
        "avg_mentions":           avg_mentions,
        "task_focus_score":       task_focus_score,
        "harmony_score":          harmony_score,
        "emoji_ratio":            emoji_ratio,
        "emotion_density":        emotion_density,
        "initiation_rate":        initiation_rate,
        "MATTR":                  mattr,
        "betweenness_centrality": betweenness,
        "response_consistency":   response_consistency,
    })
    # response_consistency NaN (insufficient replies) → 0.0 neutral.
    feats["response_consistency"] = feats["response_consistency"].fillna(0.0)
    feats = feats.reindex(columns=ALL_METRICS).fillna(0.0)
    feats.index.name = "author"
    feats = feats.reset_index()
    return feats


# ════════════════════════════════════════════════════════════════════════════
# Sanity checks (required by project instructions)
# ════════════════════════════════════════════════════════════════════════════
# Plausible value ranges for the assert-based sanity layer.
EXPECTED_RANGES = {
    "in_out_ratio":           (0.0, 50.0),
    "avg_msg_length":         (0.0, 500.0),
    "replies_sent":           (0.0, 1.0),
    "question_ratio":         (0.0, 1.0),
    "avg_mentions":           (0.0, 50.0),
    "task_focus_score":       (0.0, 1.0),
    "harmony_score":          (0.0, 1.0),
    "emoji_ratio":            (0.0, 1.0),
    "emotion_density":        (0.0, 1.0),
    "initiation_rate":        (0.0, 1.0),
    "MATTR":                  (0.0, 1.0),
    "betweenness_centrality": (0.0, 1.0),
    "response_consistency":   (-1.0, 1.0),
}


def sanity_check_features(feats, label="dataset", near_zero_var=1e-6):
    """Print feature distributions, assert ranges, flag near-zero-variance metrics."""
    print(f"\n── SANITY: feature distributions [{label}] " + "─" * 28)
    desc = feats[ALL_METRICS].describe().T[["mean", "std", "min", "max"]]
    print(desc.round(4).to_string())

    # 1) Range assertions.
    for m, (lo, hi) in EXPECTED_RANGES.items():
        vmin, vmax = feats[m].min(), feats[m].max()
        assert vmin >= lo - 1e-9 and vmax <= hi + 1e-9, (
            f"[{label}] {m} out of expected range [{lo},{hi}] → got [{vmin},{vmax}]"
        )
    print(f"  ✓ all 13 metrics within expected ranges")

    # 2) Near-zero variance flags (uninformative metric for this group).
    flagged = [m for m in ALL_METRICS if feats[m].var() <= near_zero_var]
    if flagged:
        print(f"  ⚠ NEAR-ZERO VARIANCE (uninformative here): {flagged}")
    else:
        print(f"  ✓ every metric has usable variance")
    return flagged
