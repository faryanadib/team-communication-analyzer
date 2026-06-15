"""
features_v2.py — corrected computation of the 5 INTERACTION / NETWORK metrics.

Motivation (audit of the interaction metrics):
  The *concepts* of the 13 behaviour metrics are sound, but the first
  *implementation* of the interaction-based ones in features.py was weak:

    1. parent_id is synthetic (parse_whatsapp.py) — there is NO real reply data,
       so the whole interaction graph is reconstructed from one fragile rule:
       "a message ≤10 min after the IMMEDIATELY-PREVIOUS different author = reply".
    2. The graph only links CONSECUTIVE messages (shift(1)), so in an interleaved
       group chat "who spoke right before me" ≠ "who I answered".
    3. in_out_ratio looks at a SINGLE adjacent message (got_reply = did the very
       next message answer me) — not a real received/sent count.
    4. avg_mentions uses `@\\w+`, which MISSES real WhatsApp mentions (`@⁨Name⁩`,
       the char after @ is U+2068, not \\w) and instead matches e-mail addresses.
    5. A 10–30 min "dead zone": a reply sent 10–30 min later is neither a reply
       nor an initiation — it simply vanishes, penalising slower responders.

This module keeps the SAME concept for each metric but fixes the computation:
  • mentions parsed for Slack `<@name>`, WhatsApp `@⁨name⁩`, generic `@name`
    (e-mails excluded) and mapped back to real authors → true directed edges;
  • a WINDOWED temporal model (recent distinct speakers within REPLY_WINDOW),
    not just shift(1);
  • in_out_ratio = (replies others directed at you) / (replies you sent) over the
    whole window, not a single adjacent message;
  • betweenness on the richer mention+temporal directed graph;
  • response_consistency = inverse burstiness of the corrected reply latencies.

The 8 content metrics (avg_msg_length, question_ratio, task_focus, harmony,
emoji_ratio, emotion_density, initiation_rate, MATTR) are LEFT EXACTLY as v1
computes them, so any measured change is attributable only to the 5 fixes.

Public API mirrors features.py:
  INTERACTION_METRICS · compute_interaction_metrics(df) · merge_v2(v1_feats, df)
"""
import re
import numpy as np
import pandas as pd
import networkx as nx

from features import compute_burstiness, REPLY_WINDOW_MINS  # reuse helpers/const

# the 5 metrics this module recomputes (everything else stays v1)
INTERACTION_METRICS = [
    "in_out_ratio", "replies_sent", "avg_mentions",
    "betweenness_centrality", "response_consistency",
]

# Wider window than v1's 10 min: removes the 10–30 min dead zone. A message is a
# candidate "response" if a different author spoke within this window before it.
REPLY_WINDOW_V2 = 45.0          # minutes
IN_OUT_CAP = 50.0               # clamp pathological ratios (matches EXPECTED_RANGES)

# ── mention parsing ─────────────────────────────────────────────────────────
# Slack:    <@Joey>
# WhatsApp: @⁨Alice Smith⁩   (U+2068 … U+2069 wrap the display name)
#           @~Bob Jones   /  @Alice
# generic:  @handle   (but NOT an e-mail: no word char right before @)
_SLACK_MENTION = re.compile(r"<@([^>]+)>")
_WA_MENTION    = re.compile("@[~]?⁨?([^⁩@<>\n]{1,40}?)⁩")
_AT_MENTION    = re.compile(r"(?<![\w.])@[~]?([A-Za-z0-9_][\w.\-]{0,40})")


def _raw_mentions(text):
    """All mention strings in a message (Slack, WhatsApp, generic, phone)."""
    out = []
    out += _SLACK_MENTION.findall(text)
    out += _WA_MENTION.findall(text)
    # generic @handle, skipping anything already captured as <@..> or e-mails
    cleaned = _SLACK_MENTION.sub(" ", text)
    out += _AT_MENTION.findall(cleaned)
    return [m.strip() for m in out if m.strip()]


def _build_author_matcher(authors):
    """Return fn(mention_str) -> author key or None (token / substring match)."""
    norm = {a: a.lower().strip() for a in authors}
    # first-token index for fast 'Alice Smith' -> 'Alice'
    def match(mention):
        m = mention.lower().strip()
        if not m or m in ("all", "everyone", "here", "channel"):
            return None
        # exact
        for a, an in norm.items():
            if m == an:
                return a
        # author key is a whitespace token of the mention (WhatsApp full names)
        mtokens = set(re.split(r"[\s_]+", m))
        for a, an in norm.items():
            if an in mtokens:
                return a
        # mention is a prefix of an author key or vice-versa (Slack handles)
        for a, an in norm.items():
            if an.startswith(m) or m.startswith(an):
                if min(len(an), len(m)) >= 3:
                    return a
        return None
    return match


# ── main ────────────────────────────────────────────────────────────────────
def compute_interaction_metrics(df, verbose=False):
    """
    Recompute the 5 interaction metrics on a clean message frame.
    `df`: columns author, datetime (datetime64 or ms epoch), body.
    Returns DataFrame[author, in_out_ratio, replies_sent, avg_mentions,
                      betweenness_centrality, response_consistency].
    """
    df = df.copy()
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["body"] = df["body"].fillna("").astype(str)

    authors = list(pd.unique(df["author"]))
    match = _build_author_matcher(authors)
    times = df["datetime"].to_numpy()
    auth = df["author"].tolist()

    # per-message mention targets + a raw mention count (for avg_mentions)
    mention_count = np.zeros(len(df), dtype=float)
    mention_targets = [[] for _ in range(len(df))]
    for i, body in enumerate(df["body"].tolist()):
        if "@" not in body and "<@" not in body:
            continue
        raws = _raw_mentions(body)
        mention_count[i] = len(raws)
        for r in raws:
            t = match(r)
            if t is not None and t != auth[i]:
                mention_targets[i].append(t)

    # windowed reply model -------------------------------------------------
    out_count = {a: 0 for a in authors}     # reply MESSAGES this author sent
    in_count = {a: 0 for a in authors}      # reply-acts others directed at them
    reply_latency = {a: [] for a in authors}
    total_msgs = df.groupby("author").size().to_dict()
    edges = {}                              # (src->dst) -> weight

    # index of recent messages within the window, as (time, author)
    from collections import deque
    recent = deque()
    win = np.timedelta64(int(REPLY_WINDOW_V2 * 60), "s")

    for i in range(len(df)):
        a, t = auth[i], times[i]
        # drop messages outside the window
        while recent and (t - recent[0][0]) > win:
            recent.popleft()

        targets = []
        if mention_targets[i]:                       # explicit address wins
            targets = list(dict.fromkeys(mention_targets[i]))
            # latency to the mentioned person's most recent prior message
            lat = None
            for rt, ra in reversed(recent):
                if ra in targets:
                    lat = (t - rt) / np.timedelta64(1, "m"); break
        else:
            # temporal fallback: most recent DIFFERENT author within window
            lat = None
            for rt, ra in reversed(recent):
                if ra != a:
                    targets = [ra]
                    lat = (t - rt) / np.timedelta64(1, "m")
                    break

        if targets:
            out_count[a] += 1                        # this message is a reply
            if lat is not None and lat >= 0:
                reply_latency[a].append(float(lat))
            for tgt in targets:
                in_count[tgt] += 1
                edges[(a, tgt)] = edges.get((a, tgt), 0) + 1

        recent.append((t, a))

    # assemble metrics -----------------------------------------------------
    in_out_ratio, replies_sent, response_consistency = {}, {}, {}
    for a in authors:
        snt = out_count[a]
        rcv = in_count[a]
        in_out_ratio[a] = round(min(rcv / snt, IN_OUT_CAP), 3) if snt > 0 \
            else (min(float(rcv), IN_OUT_CAP) if rcv > 0 else 0.0)
        replies_sent[a] = round(snt / total_msgs.get(a, 1), 4)
        b = compute_burstiness(reply_latency[a])
        response_consistency[a] = 0.0 if (b is None or np.isnan(b)) else float(-b)

    # betweenness on the corrected directed graph
    G = nx.DiGraph()
    G.add_nodes_from(authors)
    for (s, d), w in edges.items():
        G.add_edge(s, d, weight=w)
    btw = ({a: 0.0 for a in authors} if G.number_of_edges() == 0
           else nx.betweenness_centrality(G, normalized=True))

    avg_mentions = (pd.Series(mention_count, index=df.index)
                    .groupby(df["author"]).mean())

    out = pd.DataFrame({
        "author": authors,
        "in_out_ratio": [in_out_ratio[a] for a in authors],
        "replies_sent": [replies_sent[a] for a in authors],
        "avg_mentions": [float(avg_mentions.get(a, 0.0)) for a in authors],
        "betweenness_centrality": [float(btw.get(a, 0.0)) for a in authors],
        "response_consistency": [response_consistency[a] for a in authors],
    })
    if verbose:
        print(f"  [features_v2] {len(df):,} msgs · {len(authors):,} users · "
              f"{G.number_of_edges():,} directed edges "
              f"({sum(mention_count)>0 and int(sum(mention_count)) or 0} mentions)")
    return out


def merge_v2(v1_feats, df, verbose=False):
    """Take a v1 feature frame and overwrite ONLY the 5 interaction columns."""
    v2 = compute_interaction_metrics(df, verbose=verbose)
    merged = v1_feats.drop(columns=INTERACTION_METRICS).merge(v2, on="author", how="left")
    merged[INTERACTION_METRICS] = merged[INTERACTION_METRICS].fillna(0.0)
    return merged
