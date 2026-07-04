"""
opus_labeler.py
═══════════════
Weak-supervision labeler: sends each eligible user's messages to an Opus-class
LLM and collects soft archetype labels (probability distribution over 5 archetypes).

Design decisions:
  - ALL messages in the dataset are loaded (preserves network context for features.py).
  - Only users with >= MIN_MESSAGES are labeled (low-activity users lack behavioral signal).
  - Input to Opus: the user's OWN messages only (last MAX_MESSAGES_PER_USER, chronological).
    The full-dataset context is already captured by the 13 metrics in features.py.
  - Output is a soft label (5 floats summing to 1.0), not a hard label.
  - Results are appended to opus_labels.json line by line (crash-safe: re-running
    skips already-labeled users).
  - WhatsApp (team test set) is NEVER labeled here.

Usage:
  export LLM_API_KEY=...        # your LLM provider's API key
  export OPUS_MODEL_ID=...      # your provider's Opus-class model identifier
  python opus_labeler.py                     # label both slack + nankani
  python opus_labeler.py --dataset slack     # label only slack
  python opus_labeler.py --dry-run           # show stats, call Opus on 3 users only
  python opus_labeler.py --min-messages 30   # override threshold

Scientific grounding:
  Weak supervision via LLM oracle: Ratner et al. (2017) "Snorkel: Rapid training
  data creation with weak supervision". VLDB 2020.
  Soft labels for RF regression: soft label training reduces overconfidence
  (Hinton et al. 2015, Knowledge Distillation).
"""

import os
import sys
import json
import time
import argparse
from collections import Counter

# Per project constraint: import ONLY load_clean and ALL_METRICS from features.py
# (nothing from classify_archetypes.py → no circular import, no rule-label leakage).
from features import load_clean, ALL_METRICS
import config


# ════════════════════════════════════════════════════════════════════════════
# System prompt — the behavioral rubric Opus applies to each user's messages.
# ════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a team communication behavior analyst. Classify this person's communication
archetype based solely on their messages. Do not infer personality — infer only
observable communication behavior patterns.

The five archetypes (from Gloor 2022, Happimetrics):

BEE   — Creative connector. Starts conversations unprompted. Jumps between topics.
        Asks many questions. @mentions others to connect them. Irregular timing.
        Broad vocabulary. Acts as a bridge between people.

ANT   — Reliable executor. Task-focused language ("done", "fixed", "ready", "attached").
        Consistent, predictable response timing. Stays on one topic until resolved.
        Minimal emotional expression. Rarely starts conversations.

BUTTERFLY — Social warmth. Frequent emojis. Fast enthusiastic replies.
            Emotional affirming language. Short warm messages. Positive energy.

CAPYBARA  — Harmony keeper. Affirms before adding ("yes exactly...", "good point,
            and..."). Rarely initiates. Calm, thoughtful replies. Longer messages
            that validate others first. Conflict-avoider.

LEECH — Minimal contribution. Appears only when needing something ("can you send me",
        "where is the file"). Rarely replies to others. Short transactional messages.
        Passive, low-reciprocity presence.

RULES:
- Read the actual language and tone — not just keywords.
- Output is ALWAYS a JSON object with exactly these keys: bee, ant, butterfly,
  capybara, leech (floats summing to 1.0), confidence (high/medium/low), notes
  (one sentence max, observable behavior only, no personality judgments).
- If the messages are too few or ambiguous: confidence = "low", distribute
  scores evenly toward 0.2 for each archetype.
- Mixed profiles are valid and expected (e.g. bee=0.55, ant=0.30).

OUTPUT: valid JSON only. No prose before or after the JSON."""


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════
MIN_MESSAGES = 20
MAX_MESSAGES_PER_USER = 150
RATE_LIMIT_PAUSE = 0.5   # seconds between calls
# Model id is read from the environment so no specific vendor model id is
# committed here. Set OPUS_MODEL_ID to your provider's Opus-class model.
OPUS_MODEL = os.environ.get("OPUS_MODEL_ID", "")
OUTPUT_FILE = config.OPUS_LABELS

ARCHETYPE_KEYS = ["bee", "ant", "butterfly", "capybara", "leech"]
COST_PER_CALL = 0.015    # rough $/call estimate for Opus (Step-9 cost sanity)

# Only training-source datasets are ever labeled here. WhatsApp = held-out test set.
DATASETS = {
    "slack":   config.SLACK_CLEAN,
    "nankani": config.NANKANI_CLEAN,
    "mbada":   config.MBADA_CLEAN,
}


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_user_messages(json_path):
    """
    Read a *_clean.json (one JSON object per line) and group messages by author.

    Uses features.load_clean — the project's canonical JSONL loader — so the
    datetime parsing/ordering is byte-for-byte identical to what features.py sees
    (single source of truth, no drift). Returns {author: [msg_dict, ...]} with
    each user's messages in ascending datetime order. ALL messages are kept here
    (the >= MIN_MESSAGES filter is applied later, in label_dataset).
    """
    df = load_clean(json_path)                 # parses datetime, sorts ascending
    if "is_reply" not in df.columns:
        df["is_reply"] = 0
    by_user = {}
    for author, sub in df.groupby("author", sort=False):
        by_user[author] = sub.to_dict("records")
    return by_user


def format_for_opus(user_id, messages, max_n=MAX_MESSAGES_PER_USER):
    """
    Build the plain-text transcript shown to Opus.

    Uses the LAST `max_n` messages (most recent behavior is most representative),
    prefixes each with [R] (reply) or [I] (initiation/own-thread), and truncates
    each body to 300 characters.
    """
    recent = messages[-max_n:]
    lines = [
        f"User: {user_id} | Total messages: {len(messages)} | Showing last {len(recent)}:",
        "",
    ]
    for m in recent:
        tag = "[R]" if int(m.get("is_reply", 0)) == 1 else "[I]"
        body = str(m.get("body", "")).replace("\n", " ").strip()[:300]
        lines.append(f"{tag} {body}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# LLM provider adapter — the ONLY vendor-specific code, isolated here on purpose.
# Swap the body for your own provider's SDK. It must expose a `.messages.create(
# model, max_tokens, system, messages)` call returning an object whose
# `.content[0].text` holds the model's text reply (or adapt call_opus to match).
# ════════════════════════════════════════════════════════════════════════════
def build_llm_client(api_key):
    """Return an LLM client instance. Reads the provider SDK lazily so the rest
    of the module (and --dry-run) works without any provider package installed."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════
# Opus call + JSON validation
# ════════════════════════════════════════════════════════════════════════════
def _extract_json(text):
    """Pull the first JSON object out of a model response (tolerates ``` fences)."""
    text = text.strip()
    if text.startswith("```"):
        # drop the opening fence (``` or ```json) and the trailing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start:end + 1])


def call_opus(user_id, formatted_text, client):
    """
    Send one user's transcript to Opus and return a validated soft-label dict
    {bee, ant, butterfly, capybara, leech, confidence, notes} or None on any error.

    Validation: all 5 archetype keys present, all numeric, sum within [0.95, 1.05].
    Scores are renormalized to sum exactly 1.0 before returning. Never raises.
    """
    try:
        resp = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Classify:\n\n{formatted_text}"}],
        )
        data = _extract_json(resp.content[0].text)

        # all five archetype keys present and numeric
        vals = {}
        for k in ARCHETYPE_KEYS:
            if k not in data:
                raise ValueError(f"missing key '{k}'")
            v = data[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"key '{k}' is not a float: {v!r}")
            vals[k] = float(v)

        total = sum(vals.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"scores sum to {total:.3f}, outside [0.95, 1.05]")
        # renormalize to exactly 1.0 for clean downstream soft labels
        vals = {k: round(v / total, 4) for k, v in vals.items()}

        conf = str(data.get("confidence", "low")).lower().strip()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        notes = str(data.get("notes", "")).strip()[:300]

        return {**vals, "confidence": conf, "notes": notes}
    except Exception as e:
        print(f"    ⚠ call_opus failed for {user_id!r}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Resume support
# ════════════════════════════════════════════════════════════════════════════
def load_existing_labels(output_file):
    """Return the set of (user_id, dataset) tuples already present in output_file."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["user_id"], rec["dataset"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ════════════════════════════════════════════════════════════════════════════
# Dry-run mock (no API key / no network needed)
# ════════════════════════════════════════════════════════════════════════════
def _mock_label(messages):
    """
    Deterministic offline stand-in for call_opus used only in --dry-run.
    A tiny keyword heuristic so the dry-run output looks plausible; this is
    NEVER written to disk and NEVER used for training.
    """
    bodies = " ".join(str(m.get("body", "")).lower() for m in messages)
    raw = {
        "bee":       1.0 + bodies.count("?") * 0.2,
        "ant":       1.0 + sum(bodies.count(w) for w in ("done", "fixed", "ready")) * 0.2,
        "butterfly": 1.0 + sum(bodies.count(w) for w in ("thanks", "great", "!")) * 0.1,
        "capybara":  1.0 + sum(bodies.count(w) for w in ("agree", "exactly", "good point")) * 0.2,
        "leech":     1.0 + sum(bodies.count(w) for w in ("can you send", "where is")) * 0.5,
    }
    total = sum(raw.values())
    vals = {k: round(v / total, 4) for k, v in raw.items()}
    return {**vals, "confidence": "low", "notes": "(dry-run mock — no API call)"}


# ════════════════════════════════════════════════════════════════════════════
# Per-dataset labeling
# ════════════════════════════════════════════════════════════════════════════
def label_dataset(json_path, dataset_name, output_file, client,
                  dry_run=False, min_messages=MIN_MESSAGES):
    """
    Label every eligible (>= min_messages) user of one dataset.

    Writes one validated JSON record per user to `output_file` (append mode) the
    moment it succeeds, so an interrupted run can be resumed safely. Returns
    (n_labeled, n_failed, n_skipped) for the run summary.
    """
    print(f"\n{'═' * 64}\n  DATASET: {dataset_name}  ({json_path})\n{'═' * 64}")
    by_user = load_user_messages(json_path)
    eligible = sorted(
        (u for u, msgs in by_user.items() if len(msgs) >= min_messages),
        key=lambda u: len(by_user[u]), reverse=True,
    )
    print(f"  Users total: {len(by_user)}  |  eligible (>= {min_messages} msgs): {len(eligible)}")

    # ── Dry-run: format + mock-label the first 3 eligible users, write nothing. ─
    if dry_run:
        sample = eligible[:3]
        print(f"  DRY RUN — showing formatted input + mock labels for {len(sample)} users:\n")
        for u in sample:
            formatted = format_for_opus(u, by_user[u])
            print("  " + "─" * 60)
            print("\n".join("  " + ln for ln in formatted.splitlines()[:12]))
            if len(by_user[u]) > 11:
                print("    … (truncated)")
            lbl = _mock_label(by_user[u])
            print(f"  → mock: " + " ".join(f"{k}={lbl[k]}" for k in ARCHETYPE_KEYS)
                  + f" conf={lbl['confidence']}")
            print()
        return 0, 0, 0

    already = load_existing_labels(output_file)
    todo = [u for u in eligible if (u, dataset_name) not in already]
    skipped = len(eligible) - len(todo)
    if skipped:
        print(f"  Resume: {skipped} already labeled → {len(todo)} remaining")

    labeled = failed = 0
    with open(output_file, "a", encoding="utf-8") as out:
        for i, u in enumerate(todo, 1):
            msgs = by_user[u]
            formatted = format_for_opus(u, msgs)
            result = call_opus(u, formatted, client)
            if result is None:
                failed += 1
                continue
            record = {
                "user_id": u,
                "dataset": dataset_name,
                **{k: result[k] for k in ARCHETYPE_KEYS},
                "confidence": result["confidence"],
                "notes": result["notes"],
                "n_messages_total": len(msgs),
                "n_messages_analyzed": min(len(msgs), MAX_MESSAGES_PER_USER),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            labeled += 1
            print(f"  [{i}/{len(todo)}] {str(u)[:18]:<18} | "
                  + " ".join(f"{k}={result[k]:.2f}" for k in ARCHETYPE_KEYS)
                  + f" conf={result['confidence']}")
            time.sleep(RATE_LIMIT_PAUSE)

    print(f"\n  {dataset_name} summary → labeled {labeled}, failed {failed}, skipped {skipped}")
    return labeled, failed, skipped


# ════════════════════════════════════════════════════════════════════════════
# Post-labeling sanity checks (Step 9)
# ════════════════════════════════════════════════════════════════════════════
def sanity_check_output(output_file):
    """Distribution of dominant archetypes, confidence breakdown, coverage + cost."""
    if not os.path.exists(output_file):
        print(f"\n  (no {output_file} to summarize)")
        return
    records = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    if n == 0:
        print(f"\n  {output_file} is empty.")
        return

    print(f"\n{'═' * 64}\n  SANITY CHECKS — {output_file}  ({n} labeled users)\n{'═' * 64}")

    # Dominant archetype distribution.
    dom = Counter(max(ARCHETYPE_KEYS, key=lambda k: r[k]) for r in records)
    print("  Dominant archetype distribution:")
    for k in ARCHETYPE_KEYS:
        c = dom.get(k, 0)
        print(f"    {k:<10} {c:>5}  ({c / n * 100:5.1f}%)  {'█' * int(c / n * 40)}")

    # Mean confidence breakdown.
    conf = Counter(r.get("confidence", "low") for r in records)
    print("  Confidence breakdown:")
    for level in ("high", "medium", "low"):
        c = conf.get(level, 0)
        print(f"    {level:<8} {c:>5}  ({c / n * 100:5.1f}%)")

    # Warn on under-represented archetypes (< 5% of labeled users).
    rare = [k for k in ARCHETYPE_KEYS if dom.get(k, 0) / n < 0.05]
    if rare:
        print(f"  ⚠ archetypes under 5% of labeled users: {rare}")
    else:
        print("  ✓ every archetype is the dominant label for >= 5% of users")

    # Estimated cost.
    print(f"  Estimated cost: {n} calls × ${COST_PER_CALL:.3f} ≈ ${n * COST_PER_CALL:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Opus weak-supervision archetype labeler.")
    parser.add_argument("--dataset", choices=["slack", "nankani", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="format + mock-label 3 users per dataset; no API calls, no writes")
    parser.add_argument("--min-messages", type=int, default=MIN_MESSAGES)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    print("\n🧭  SocialCompass — Opus weak-supervision labeler")
    print("=" * 64)
    print(f"  features.py captures {len(ALL_METRICS)} network/linguistic metrics separately;")
    print("  Opus here reads only the RAW TEXT to avoid circular labeling.")

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    # ── Build the client (skipped entirely for dry runs). ───────────────────
    client = None
    if not args.dry_run:
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            print("\n  ✗ LLM_API_KEY is not set.")
            print("    export LLM_API_KEY=...   then re-run.")
            sys.exit(1)
        if not OPUS_MODEL:
            print("\n  ✗ OPUS_MODEL_ID is not set.")
            print("    export OPUS_MODEL_ID=<your provider's model id>   then re-run.")
            sys.exit(1)
        client = build_llm_client(api_key)

    totals = [0, 0, 0]   # labeled, failed, skipped
    for name in targets:
        path = DATASETS[name]
        if not os.path.exists(path):
            print(f"\n  ⚠ skipping {name}: {path} not found")
            continue
        l, f, s = label_dataset(path, name, args.output, client,
                                dry_run=args.dry_run, min_messages=args.min_messages)
        totals = [totals[0] + l, totals[1] + f, totals[2] + s]

    if args.dry_run:
        print("\n  Dry run complete — no API calls made")
        return

    print(f"\n{'═' * 64}")
    print(f"  RUN TOTAL → labeled {totals[0]}, failed {totals[1]}, skipped {totals[2]}")
    sanity_check_output(args.output)
    print(f"\n  ✅ Labels written → {args.output}")


if __name__ == "__main__":
    main()
