import pandas as pd
import re
import config

# ── Load your exported WhatsApp .txt file ────────────────────────────────
WHATSAPP_FILE = config.WHATSAPP_RAW

# ── Map raw WhatsApp author labels → clean display names ─────────────────
# WhatsApp exports show full display names (e.g. "~ Alice Smith",
# "Bob Some Suffix"), so we match on a SUBSTRING of the lower-cased author
# rather than an exact phone number. Customize this map for your own chat —
# leave it empty to keep every author's raw name as-is.
NAME_SUBSTR_MAP = {
    # 'alice': 'Alice',
    # 'bob'  : 'Bob',
}

# Authors that are really the GROUP itself / system, not a person → dropped.
# (WhatsApp posts "You created this group", encryption notice, etc. under the
#  group's own name.) Add your group's title here so it is not counted as a user.
GROUP_AUTHOR_MARKERS = [
    # 'my group title',
]


def normalize_author(raw):
    """Return a clean first name, or None if the author is the group/system."""
    a = raw.strip().lstrip('~').strip()          # drop WhatsApp "~ " non-contact prefix
    low = a.lower()
    if any(m in low for m in GROUP_AUTHOR_MARKERS):
        return None                              # the group pseudo-user → drop
    for key, name in NAME_SUBSTR_MAP.items():
        if key in low:
            return name
    return a                                     # unknown but real → keep as-is

def parse_whatsapp(filepath):
    """Parse WhatsApp export into a clean DataFrame."""

    pattern = re.compile(
        r'[‎\[]?'                                         # optional LTR mark or [
        r'(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}),?\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)'
        r'[\]\s]*[-–]?\s*'                                    # ] or space then optional dash
        r'([^:]+?):\s(.+)'
    )

    messages    = []
    current_msg = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line  = line.strip()
        match = pattern.match(line)

        if match:
            # Save previous message before starting new one
            if current_msg:
                messages.append(current_msg)

            date_str, time_str, author, body = match.groups()

            # ── Apply name mapping; skip the group/system pseudo-author ───
            author_clean = normalize_author(author)
            if author_clean is None:
                current_msg = None          # group/system line → ignore entirely
                continue

            current_msg = {
                'date_str': date_str,
                'time_str': time_str,
                'author'  : author_clean,
                'body'    : body.strip()
            }
        else:
            # Continuation of a multi-line message
            if current_msg and line:
                current_msg['body'] += ' ' + line

    # Don't forget the last message
    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)

    if len(df) == 0:
        print("❌ No messages parsed. Check your file format.")
        return df

    # ── Parse datetime ────────────────────────────────────────────────────
    df['datetime'] = pd.to_datetime(
        df['date_str'] + ' ' + df['time_str'],
        dayfirst=True,
        errors='coerce'
    )

    # ── Drop system messages ──────────────────────────────────────────────
    system_phrases = [
        'messages and calls are end-to-end encrypted',
        'changed the subject', 'added you', 'left',
        'changed this group', 'joined using this group',
        'you were added', 'image omitted', 'video omitted',
        'audio omitted', 'document omitted', 'sticker omitted',
        '<media omitted>', 'null',
        'you created this group', 'created group', 'this message was deleted',
        'you deleted this message', 'changed their phone number',
        'changed to a new phone number', 'you joined', 'tap to learn more',
    ]
    mask = df['body'].str.lower().apply(
        lambda x: not any(p in x for p in system_phrases)
    )
    df = df[mask]

    # ── Drop very short messages and bad rows ─────────────────────────────
    df = df[df['body'].str.len() >= 2]
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # ── Detect replies (within 10 mins from a different person) ──────────
    df['prev_time']   = df['datetime'].shift(1)
    df['prev_author'] = df['author'].shift(1)
    df['gap_mins']    = (
        df['datetime'] - df['prev_time']
    ).dt.total_seconds() / 60

    df['is_reply'] = (
        (df['gap_mins'] <= 10) &
        (df['prev_author'] != df['author'])
    ).astype(int)

    df['parent_id'] = df.apply(
        lambda r: f"t1_approx_{r.name - 1}"
                  if r['is_reply'] == 1
                  else f"t3_post_{r.name}",
        axis=1
    )

    # ── Final columns ─────────────────────────────────────────────────────
    df['subreddit'] = 'whatsapp_group'
    df['score']     = 1

    df = df[['author', 'datetime', 'body', 'parent_id',
             'is_reply', 'subreddit', 'score']]

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"✅ Parsed {len(df)} messages")
    print(f"👥 Unique users: {df['author'].nunique()}")
    print(f"📅 Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"↩️  Reply rate: {df['is_reply'].mean() * 100:.1f}%")
    print(f"\nUsers found:")
    print(df['author'].value_counts())
    print(f"\nSample:")
    print(df[['author', 'datetime', 'body', 'is_reply']].head(5))

    return df


df = parse_whatsapp(WHATSAPP_FILE)

if len(df) > 0:
    df.to_json(config.WHATSAPP_CLEAN, orient="records", lines=True)
    print(f"\n✅ Saved to {config.WHATSAPP_CLEAN}")