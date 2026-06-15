import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob
import config

# ── Point to the downloaded data folder ──────────────────────────────────
DATA_ROOT = os.path.join(config.SLACK_RAW_DIR, "data")

all_messages = []

# Walk through all XML files
xml_files = glob.glob(f"{DATA_ROOT}/**/*.xml", recursive=True)
print(f"Found {len(xml_files)} XML files")

for filepath in xml_files:
    # Get channel name from folder structure
    parts   = filepath.replace("\\", "/").split("/")
    channel = parts[-3] if len(parts) >= 3 else "unknown"

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for msg in root.findall('message'):
            ts   = msg.find('ts')
            user = msg.find('user')
            text = msg.find('text')
            conv = msg.get('conversation_id', '0')

            if ts is not None and user is not None and text is not None:
                all_messages.append({
                    'author'         : user.text.strip(),
                    'datetime_str'   : ts.text.strip(),
                    'body'           : text.text.strip() if text.text else '',
                    'conversation_id': conv,
                    'channel'        : channel
                })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

df = pd.DataFrame(all_messages)
print(f"\nRaw messages loaded: {len(df)}")

# ── Parse datetime ────────────────────────────────────────────────────────
df['datetime'] = pd.to_datetime(df['datetime_str'], errors='coerce')
df = df.dropna(subset=['datetime'])

# ── Clean ─────────────────────────────────────────────────────────────────
df = df[df['body'].str.len() >= 3]
df = df[~df['author'].isin(['', 'nan', 'None'])]
df = df.sort_values('datetime').reset_index(drop=True)

# ── Detect replies: same conversation_id = reply chain ───────────────────
df['prev_author'] = df.groupby('conversation_id')['author'].shift(1)
df['is_reply']    = (
    df['prev_author'].notna() &
    (df['prev_author'] != df['author'])
).astype(int)

# ── Sample 60k messages for manageability ────────────────────────────────
if len(df) > 60000:
    # Keep users with at least 3 messages for meaningful features
    active_users = df['author'].value_counts()
    active_users = active_users[active_users >= 3].index
    df = df[df['author'].isin(active_users)]
    df = df.head(60000)

df['parent_id'] = df.apply(
    lambda r: f"t1_{r.name-1}" if r['is_reply'] else f"t3_{r.name}",
    axis=1
)

print(f"Clean messages: {len(df)}")
print(f"Unique users: {df['author'].nunique()}")
print(f"Channels: {df['channel'].value_counts().to_dict()}")
print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Reply rate: {df['is_reply'].mean()*100:.1f}%")
print(f"\nTop 10 most active users:")
print(df['author'].value_counts().head(10))

df.to_json(config.SLACK_CLEAN, orient="records", lines=True)
print(f"\n✅ Saved to {config.SLACK_CLEAN}")