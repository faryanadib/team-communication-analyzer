"""
build_annotation_sample.py
══════════════════════════
Prepare the data + tooling for the HUMAN inter-rater validation study of the
Opus soft labels (opus_labels.json).

Why this exists
---------------
The 300 soft labels are produced by a single LLM annotator (Opus). To claim they
are scientifically trustworthy we run a small inter-annotator-agreement (IAA)
study: independent human annotators re-label a stratified sample *blind* to the
model, and we report whether human↔Opus agreement is comparable to human↔human
agreement (the human reliability ceiling). Humans are NOT treated as ground
truth — they are a second noisy annotator; the claim is "Opus operates within
human inter-rater reliability", which justifies using it as a weak supervisor.

What this script does
---------------------
1. Loads opus_labels.json and keeps ONLY the 247 real, Opus-labeled users
   (slack / nankani / mbada). The 53 `synthetic` users were hand-labeled by
   reading (label_synthetic.py), NOT by Opus — including them would validate our
   own hand labels, not Opus, so they are excluded.
2. Stratified random sample of `--per-class` users per dominant archetype
   (default 10) so rare classes (butterfly=16, capybara=19) are covered.
3. Reconstructs each sampled user's transcript EXACTLY as Opus saw it: the last
   150 messages, each tagged [I] (initiation) / [R] (reply) and truncated to
   300 chars — byte-for-byte parity with opus_labeler.format_for_opus, so the
   human reads the same input the model read (fair comparison).
4. Writes:
     outputs/annotation/sample_users.json          (blinded — no labels)
     outputs/annotation/answer_key_DO_NOT_SHARE.json (Opus labels, hidden)
     outputs/annotation/assignment.json            (study design metadata)
     outputs/annotation/annotator_<name>/label_app.html  (self-contained app)
     outputs/annotation/annotator_<name>/READ_ME_FIRST.md
   Each annotator gets a FULL copy of the sample (full overlap → we can measure
   both human↔human and human↔Opus on the same users), shown in an independently
   shuffled order to dampen order/fatigue effects.

The annotation task (decided with the team)
-------------------------------------------
For each user the annotator splits 100 points across the 5 archetypes (directly
comparable to the Opus soft label and the project's Σmin soft-accuracy metric)
and ticks a confidence (High / Medium / Low). The app enforces sum = 100 and
exports a CSV that opens straight in Excel.

Usage
-----
    python src/build_annotation_sample.py                  # 10/class, annotators 1 & 2
    python src/build_annotation_sample.py --per-class 8
    python src/build_annotation_sample.py --annotators Sara Reza Ali
    python src/build_annotation_sample.py --seed 7

Scientific grounding: Artstein & Poesio (2008) "Inter-coder agreement for
computational linguistics"; Krippendorff (2004) Content Analysis. LLM-as-
annotator validation: e.g. Gilardi et al. (2023) PNAS.
"""

import os
import json
import html
import random
import argparse
from collections import Counter, defaultdict

import pandas as pd
import config

ARCHETYPE_KEYS = ["bee", "ant", "butterfly", "capybara", "leech"]
MAX_MESSAGES_PER_USER = 150   # parity with opus_labeler.MAX_MESSAGES_PER_USER
BODY_TRUNC = 300              # parity with opus_labeler.format_for_opus
REAL_DATASETS = {"slack", "nankani", "mbada"}   # Opus-labeled; synthetic excluded

# Friendly metadata for the cheat-sheet shown to annotators. Definitions are a
# faithful condensation of the exact rubric Opus was given (opus_labeler.SYSTEM_PROMPT).
ARCHETYPE_INFO = {
    "bee":       {"emoji": "🐝", "title": "Bee — Creative connector",
                  "desc": "Starts conversations unprompted, jumps between topics, asks many questions, @mentions others to connect them, irregular timing, broad vocabulary, acts as a bridge."},
    "ant":       {"emoji": "🐜", "title": "Ant — Reliable executor",
                  "desc": "Task-focused language (done / fixed / ready / attached), consistent predictable timing, stays on one topic until resolved, minimal emotion, rarely initiates."},
    "butterfly": {"emoji": "🦋", "title": "Butterfly — Social warmth",
                  "desc": "Frequent emojis, fast enthusiastic replies, affirming/emotional language, short warm messages, positive energy."},
    "capybara":  {"emoji": "🦫", "title": "Capybara — Harmony keeper",
                  "desc": "Affirms before adding (\"yes exactly, and...\"), rarely initiates, calm thoughtful longer replies that validate others first, conflict-avoider."},
    "leech":     {"emoji": "🔴", "title": "Leech — Minimal contribution",
                  "desc": "Appears only when needing something (\"can you send me\", \"where is the file\"), rarely replies, short transactional messages, passive, low-reciprocity."},
}

REMINDERS = [
    "Judge only from the text. Don't guess personality — rate observable communication behavior.",
    "A mixed profile is normal (e.g. bee=50, ant=30). You don't have to give everything to one.",
    "The 5 numbers must sum to exactly 100 (the counter above the form helps).",
    "[I] = a message that starts a conversation · [R] = a reply to someone else.",
    "If messages are few or ambiguous → set confidence to Low and spread points near-evenly (~20 each).",
    "You never see any model's or anyone else's labels — this is intentional so your judgment stays independent.",
]


# ════════════════════════════════════════════════════════════════════════════
# Data loading (lightweight — no model imports, byte-parity with opus_labeler)
# ════════════════════════════════════════════════════════════════════════════
def load_clean_grouped(json_path):
    """Read a *_clean.json (JSONL) → {author: [msg_dict, ...]} sorted ascending.

    Mirrors features.load_clean (sort by datetime ms) + opus_labeler grouping,
    but without importing features/ml_features (avoids loading RoBERTa for a
    pure data-prep step). is_reply is taken from the clean file as-is.
    """
    df = pd.read_json(json_path, lines=True)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["body"] = df["body"].fillna("").astype(str)
    if "is_reply" not in df.columns:
        df["is_reply"] = 0
    by_user = {}
    for author, sub in df.groupby("author", sort=False):
        by_user[author] = sub.to_dict("records")
    return by_user


def build_transcript(messages):
    """Return the structured transcript Opus saw: last 150 msgs, [I]/[R] tag,
    body truncated to 300 chars. List of {"t": "I"|"R", "b": str}."""
    recent = messages[-MAX_MESSAGES_PER_USER:]
    out = []
    for m in recent:
        tag = "R" if int(m.get("is_reply", 0) or 0) == 1 else "I"
        body = str(m.get("body", "")).replace("\n", " ").strip()[:BODY_TRUNC]
        if body:
            out.append({"t": tag, "b": body})
    return out


def dominant(rec):
    return max(ARCHETYPE_KEYS, key=lambda k: rec[k])


# ════════════════════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════════════════════
def stratified_sample(records, per_class, rng):
    """Pick `per_class` users per dominant archetype (or all if fewer exist)."""
    by_arch = defaultdict(list)
    for r in records:
        by_arch[dominant(r)].append(r)
    picked = []
    summary = {}
    for k in ARCHETYPE_KEYS:
        pool = by_arch.get(k, [])
        rng.shuffle(pool)
        take = pool[: min(per_class, len(pool))]
        picked.extend(take)
        summary[k] = {"available": len(pool), "sampled": len(take)}
    return picked, summary


# ════════════════════════════════════════════════════════════════════════════
# HTML app generation (self-contained, data embedded)
# ════════════════════════════════════════════════════════════════════════════
def _safe_json_for_script(obj):
    """json.dumps safe to embed inside a <script> tag."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_app(annotator, cards, cheatsheet, reminders):
    """Build a single self-contained HTML annotation app for one annotator."""
    data_json = _safe_json_for_script(cards)
    cheat_json = _safe_json_for_script(cheatsheet)
    rem_json = _safe_json_for_script(reminders)
    keys_json = _safe_json_for_script(ARCHETYPE_KEYS)
    return (HTML_TEMPLATE
            .replace("__ANNOTATOR__", html.escape(str(annotator)))
            .replace("__DATA_JSON__", data_json)
            .replace("__CHEAT_JSON__", cheat_json)
            .replace("__REM_JSON__", rem_json)
            .replace("__KEYS_JSON__", keys_json))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Archetype labeling — Annotator __ANNOTATOR__</title>
<style>
  :root{ --bg:#0f1220; --card:#1a1e33; --ink:#e8ebf5; --muted:#9aa3c4;
         --accent:#6ea8fe; --good:#43c59e; --bad:#ef6f6c; --line:#2a2f4a; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--ink);
        font-family:Vazirmatn,Tahoma,system-ui,sans-serif; }
  header{ position:sticky; top:0; z-index:10; background:#12152a; border-bottom:1px solid var(--line);
          padding:10px 16px; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1{ font-size:15px; margin:0; font-weight:700; }
  .pill{ background:var(--card); border:1px solid var(--line); border-radius:999px;
         padding:4px 12px; font-size:12px; color:var(--muted); }
  .wrap{ display:grid; grid-template-columns: 1fr 360px; gap:16px; padding:16px; max-width:1280px; margin:0 auto; }
  @media(max-width:980px){ .wrap{ grid-template-columns:1fr; } }
  .panel{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; }
  .uid{ font-size:14px; font-weight:700; }
  .meta{ font-size:12px; color:var(--muted); margin-top:2px; }
  .transcript{ margin-top:10px; max-height:60vh; overflow:auto; direction:ltr; text-align:left;
               border:1px solid var(--line); border-radius:10px; padding:8px; background:#0d1020; }
  .msg{ padding:5px 8px; border-bottom:1px solid #1c2138; font-size:13px; line-height:1.5; }
  .msg:last-child{ border-bottom:none; }
  .tag{ display:inline-block; min-width:24px; font-weight:700; font-size:11px; margin-inline-end:6px; }
  .tag.I{ color:var(--accent); } .tag.R{ color:var(--good); }
  .side h3{ margin:0 0 8px; font-size:13px; }
  .arch{ border:1px solid var(--line); border-radius:10px; padding:8px 10px; margin-bottom:8px; }
  .arch b{ font-size:13px; } .arch p{ margin:4px 0 0; font-size:12px; color:var(--muted); line-height:1.6; }
  .rem{ font-size:12px; color:var(--muted); line-height:1.9; }
  .rem li{ margin-bottom:4px; }
  .scorebox{ margin-top:8px; }
  .row{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }
  .row label{ width:120px; font-size:13px; }
  .row input[type=range]{ flex:1; }
  .row input[type=number]{ width:64px; background:#0d1020; color:var(--ink);
        border:1px solid var(--line); border-radius:8px; padding:6px; text-align:center; font-size:14px; }
  .sum{ font-weight:700; padding:6px 12px; border-radius:8px; display:inline-block; }
  .sum.ok{ background:rgba(67,197,158,.15); color:var(--good); }
  .sum.bad{ background:rgba(239,111,108,.15); color:var(--bad); }
  .conf{ display:flex; gap:8px; margin-top:10px; }
  .conf label{ flex:1; text-align:center; border:1px solid var(--line); border-radius:8px;
        padding:8px; cursor:pointer; font-size:13px; }
  .conf input{ display:none; }
  .conf input:checked + span{ font-weight:700; }
  .conf label:has(input:checked){ border-color:var(--accent); background:rgba(110,168,254,.12); }
  textarea{ width:100%; margin-top:10px; background:#0d1020; color:var(--ink);
        border:1px solid var(--line); border-radius:8px; padding:8px; font-family:inherit; font-size:13px; }
  .nav{ display:flex; gap:8px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button{ background:var(--accent); color:#08111f; border:none; border-radius:9px;
        padding:9px 14px; font-size:14px; font-weight:700; cursor:pointer; }
  button.ghost{ background:transparent; color:var(--ink); border:1px solid var(--line); }
  button:disabled{ opacity:.4; cursor:not-allowed; }
  .done{ color:var(--good); } .todo{ color:var(--muted); }
  select{ background:#0d1020; color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:6px; }
  .bar{ height:6px; background:#0d1020; border-radius:999px; overflow:hidden; flex:1; min-width:120px; }
  .bar > i{ display:block; height:100%; background:var(--good); width:0%; }
  .save-note{ font-size:11px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Communication-archetype labeling</h1>
  <span class="pill">Annotator: <b id="who">__ANNOTATOR__</b></span>
  <span class="pill"><span id="progress">0</span> of <span id="total">0</span> saved</span>
  <div class="bar" style="max-width:200px"><i id="barfill"></i></div>
  <span class="pill">User <span id="cardidx">1</span></span>
  <button class="ghost" id="exportBtn">⬇️ Download CSV</button>
</header>

<div class="wrap">
  <!-- LEFT: transcript + scoring -->
  <div>
    <div class="panel">
      <div class="uid" id="uid">—</div>
      <div class="meta" id="meta">—</div>
      <div class="transcript" id="transcript"></div>
    </div>

    <div class="panel" style="margin-top:14px">
      <h3 style="margin:0 0 4px">Split 100 points across the 5 archetypes</h3>
      <div class="meta">Rate communication behavior, not personality. A mixed profile is perfectly fine.</div>
      <div class="scorebox" id="scorebox"></div>
      <div style="margin-top:6px">
        Sum: <span class="sum bad" id="sum">0</span>
        <button class="ghost" id="normBtn" style="margin-inline-start:8px">↺ Normalize to 100</button>
      </div>

      <div class="conf" id="conf">
        <label><input type="radio" name="conf" value="high"><span>High confidence</span></label>
        <label><input type="radio" name="conf" value="medium"><span>Medium</span></label>
        <label><input type="radio" name="conf" value="low"><span>Low</span></label>
      </div>

      <textarea id="note" rows="2" placeholder="Optional note (one sentence — observed behavior)"></textarea>

      <div class="nav">
        <button class="ghost" id="prevBtn">◀ Prev</button>
        <button id="saveBtn">Save &amp; next ▶</button>
        <select id="jump"></select>
        <span class="save-note" id="savestate">Auto-save is on</span>
      </div>
    </div>
  </div>

  <!-- RIGHT: cheat-sheet + reminders -->
  <div class="side">
    <div class="panel">
      <h3>Archetype reminder</h3>
      <div id="cheat"></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Tips</h3>
      <ul class="rem" id="reminders"></ul>
    </div>
  </div>
</div>

<script type="application/json" id="cards">__DATA_JSON__</script>
<script type="application/json" id="cheatdata">__CHEAT_JSON__</script>
<script type="application/json" id="remdata">__REM_JSON__</script>
<script type="application/json" id="keysdata">__KEYS_JSON__</script>
<script>
const ANNOTATOR = "__ANNOTATOR__";
const CARDS = JSON.parse(document.getElementById('cards').textContent);
const CHEAT = JSON.parse(document.getElementById('cheatdata').textContent);
const REM   = JSON.parse(document.getElementById('remdata').textContent);
const KEYS  = JSON.parse(document.getElementById('keysdata').textContent);
const LS_KEY = "annot_" + ANNOTATOR;
let store = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
let idx = 0;
let cardStart = Date.now();

// ---- static panels ----
function renderCheat(){
  document.getElementById('cheat').innerHTML = CHEAT.map(c =>
    `<div class="arch"><b>${c.emoji} ${c.title}</b><p>${c.desc}</p></div>`).join('');
  document.getElementById('reminders').innerHTML = REM.map(r => `<li>${r}</li>`).join('');
  document.getElementById('total').textContent = CARDS.length;
  const jump = document.getElementById('jump');
  jump.innerHTML = CARDS.map((c,i)=>`<option value="${i}">${i+1}. ${c.uid}</option>`).join('');
}

function recById(){ return store[CARDS[idx].cid] || null; }

function renderCard(){
  const c = CARDS[idx];
  cardStart = Date.now();
  document.getElementById('cardidx').textContent = (idx+1);
  document.getElementById('jump').value = idx;
  document.getElementById('uid').textContent = c.uid;
  document.getElementById('meta').textContent =
    `Source: ${c.dataset} · messages shown: ${c.transcript.length} (of ${c.n_total})`;
  document.getElementById('transcript').innerHTML = c.transcript.map(m =>
    `<div class="msg"><span class="tag ${m.t}">[${m.t}]</span>${escapeHtml(m.b)}</div>`).join('');

  // scoring rows
  const saved = recById();
  const sb = document.getElementById('scorebox');
  sb.innerHTML = KEYS.map(k=>{
    const v = saved ? saved.scores[k] : 0;
    const info = CHEAT.find(x=>x.key===k);
    return `<div class="row">
      <label>${info.emoji} ${k}</label>
      <input type="range" min="0" max="100" value="${v}" data-k="${k}" class="rng">
      <input type="number" min="0" max="100" value="${v}" data-k="${k}" class="num">
    </div>`;
  }).join('');
  sb.querySelectorAll('.rng').forEach(el=>el.oninput=()=>syncFrom(el,'num'));
  sb.querySelectorAll('.num').forEach(el=>el.oninput=()=>syncFrom(el,'rng'));

  // confidence + note
  document.querySelectorAll('input[name=conf]').forEach(r=> r.checked = saved && saved.confidence===r.value);
  document.getElementById('note').value = saved ? (saved.note||"") : "";

  updateSum();
  updateProgress();
  document.getElementById('prevBtn').disabled = (idx===0);
}

function syncFrom(el, otherClass){
  const k = el.dataset.k;
  document.querySelector(`.${otherClass}[data-k="${k}"]`).value = el.value;
  updateSum();
}

function currentScores(){
  const s = {};
  document.querySelectorAll('.num').forEach(el=> s[el.dataset.k] = Math.max(0, Math.min(100, parseInt(el.value||0))));
  return s;
}
function sumScores(s){ return KEYS.reduce((a,k)=>a+(s[k]||0),0); }

function updateSum(){
  const total = sumScores(currentScores());
  const el = document.getElementById('sum');
  el.textContent = total;
  el.className = "sum " + (total===100 ? "ok":"bad");
}

function normalize(){
  const s = currentScores(); const t = sumScores(s);
  if(t===0){ KEYS.forEach(k=>setScore(k,20)); updateSum(); return; }
  let scaled = KEYS.map(k=> [k, s[k]/t*100]);
  // round to ints summing to exactly 100 (largest-remainder)
  let floored = scaled.map(([k,v])=>[k, Math.floor(v), v-Math.floor(v)]);
  let rem = 100 - floored.reduce((a,x)=>a+x[1],0);
  floored.sort((a,b)=>b[2]-a[2]);
  for(let i=0;i<rem;i++) floored[i][1]++;
  floored.forEach(([k,v])=> setScore(k,v));
  updateSum();
}
function setScore(k,v){
  document.querySelector(`.num[data-k="${k}"]`).value = v;
  document.querySelector(`.rng[data-k="${k}"]`).value = v;
}

function saveCurrent(silent){
  const scores = currentScores();
  const total = sumScores(scores);
  const conf = (document.querySelector('input[name=conf]:checked')||{}).value || "";
  if(!silent){
    if(total!==100){ alert("Scores must sum to exactly 100 (now "+total+"). Use 'Normalize to 100'."); return false; }
    if(!conf){ alert("Please choose a confidence level (High / Medium / Low)."); return false; }
  }
  const c = CARDS[idx];
  const prev = store[c.cid];
  store[c.cid] = {
    uid: c.uid, dataset: c.dataset, scores, confidence: conf,
    note: document.getElementById('note').value.trim(),
    seconds: (prev? prev.seconds:0) + Math.round((Date.now()-cardStart)/1000),
    ts: new Date().toISOString(), complete: (total===100 && !!conf)
  };
  localStorage.setItem(LS_KEY, JSON.stringify(store));
  flashSaved();
  updateProgress();
  return true;
}

function updateProgress(){
  const done = Object.values(store).filter(r=>r.complete).length;
  document.getElementById('progress').textContent = done;
  document.getElementById('barfill').style.width = (done/CARDS.length*100)+"%";
}
function flashSaved(){
  const el = document.getElementById('savestate');
  el.textContent = "✓ Saved"; el.className="save-note done";
  setTimeout(()=>{ el.textContent="Auto-save is on"; el.className="save-note"; }, 1200);
}

function next(){ if(idx<CARDS.length-1){ idx++; renderCard(); } }
function prev(){ if(idx>0){ saveCurrent(true); idx--; renderCard(); } }

function exportCSV(){
  const header = ["annotator","user_id","dataset",...KEYS,"confidence","note","seconds","timestamp"];
  const rows = [header.join(",")];
  CARDS.forEach(c=>{
    const r = store[c.cid];
    if(!r || !r.complete) return;
    const vals = [ANNOTATOR, c.uid, c.dataset, ...KEYS.map(k=>r.scores[k]),
                  r.confidence, (r.note||"").replace(/"/g,'""'), r.seconds, r.ts];
    rows.push(vals.map(csvCell).join(","));
  });
  if(rows.length===1){ alert("No completed users yet."); return; }
  const blob = new Blob(["﻿"+rows.join("\n")], {type:"text/csv;charset=utf-8;"});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "labels_annotator_"+ANNOTATOR+".csv";
  a.click();
}
function csvCell(v){ v=String(v); return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; }
function escapeHtml(s){ return s.replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

document.getElementById('saveBtn').onclick = ()=>{ if(saveCurrent(false)) next(); };
document.getElementById('prevBtn').onclick = prev;
document.getElementById('normBtn').onclick = normalize;
document.getElementById('exportBtn').onclick = exportCSV;
document.getElementById('jump').onchange = (e)=>{ saveCurrent(true); idx=parseInt(e.target.value); renderCard(); };
window.addEventListener('beforeunload', ()=> saveCurrent(true));

renderCheat();
renderCard();
</script>
</body>
</html>
"""


def write_readme(folder, annotator, n_cards):
    txt = f"""# Annotator guide — {annotator}

Hi! Thanks for helping validate this project scientifically. 🙏

## What is this?
We randomly picked {n_cards} users. For each user you read ONLY **their messages**
and judge how much their communication behavior matches each "archetype".

## How?
1. Double-click **`label_app.html`** to open it in a browser (Chrome / Edge / Safari).
   No installation, no internet needed.
2. For each user, split **100 points** across the 5 archetypes (must sum to exactly
   100 — the "Normalize to 100" button helps).
3. Tick a **confidence** level (High / Medium / Low).
4. Click "Save & next". The archetype reminder and tips stay on the right the whole time.

## Important
- Work **independently** — don't discuss the users with the other annotator
  (we are measuring independent agreement).
- Your user order differs from the other annotator's — that's intentional.
- Your work auto-saves in the browser, but **when you finish, click
  "⬇️ Download CSV"** and send me the file `labels_annotator_{annotator}.csv`.

## Why no answer is shown
We deliberately show you no pre-made labels (neither the model's nor anyone else's)
so your judgment stays fully independent and valid. Your output is later compared
with the model's and with the other annotator's.
"""
    with open(os.path.join(folder, "READ_ME_FIRST.md"), "w", encoding="utf-8") as f:
        f.write(txt)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Build the human inter-rater validation sample + apps.")
    ap.add_argument("--per-class", type=int, default=10, help="users sampled per dominant archetype")
    ap.add_argument("--annotators", nargs="+", default=["1", "2"], help="annotator names/ids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ── load Opus labels, keep only the real (Opus-labeled) users ──────────────
    records = [json.loads(l) for l in open(config.OPUS_LABELS, encoding="utf-8") if l.strip()]
    real = [r for r in records if r["dataset"] in REAL_DATASETS]
    print(f"Opus labels: {len(records)} total → {len(real)} real (synthetic excluded).")

    picked, summ = stratified_sample(real, args.per_class, rng)
    print(f"\nStratified sample ({args.per_class}/class), seed={args.seed}:")
    for k in ARCHETYPE_KEYS:
        print(f"  {k:<10} sampled {summ[k]['sampled']:>2} / available {summ[k]['available']}")
    print(f"  TOTAL sampled: {len(picked)}")

    # ── reconstruct transcripts (cache clean files) ────────────────────────────
    clean_cache = {}
    def transcripts_for(ds):
        if ds not in clean_cache:
            clean_cache[ds] = load_clean_grouped(config.CLEAN_FILES[ds])
        return clean_cache[ds]

    cards, answer_key = [], []
    missing = []
    for i, r in enumerate(picked):
        uid, ds = r["user_id"], r["dataset"]
        by_user = transcripts_for(ds)
        msgs = by_user.get(uid)
        if not msgs:
            missing.append((uid, ds))
            continue
        cid = f"c{i:03d}"
        cards.append({
            "cid": cid, "uid": uid, "dataset": ds,
            "n_total": len(msgs),
            "transcript": build_transcript(msgs),
        })
        answer_key.append({
            "cid": cid, "user_id": uid, "dataset": ds,
            **{k: r[k] for k in ARCHETYPE_KEYS},
            "opus_dominant": dominant(r),
            "opus_confidence": r.get("confidence"),
            "opus_notes": r.get("notes", ""),
        })
    if missing:
        print(f"\n  ⚠ {len(missing)} sampled users had no transcript (skipped): {missing[:5]}")

    # ── write shared artifacts ─────────────────────────────────────────────────
    with open(config.ANNOTATION_SAMPLE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=1)
    with open(config.ANNOTATION_ANSWER_KEY, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=1)

    cheatsheet = [{"key": k, **ARCHETYPE_INFO[k]} for k in ARCHETYPE_KEYS]

    assignment = {
        "design": "full overlap — every annotator labels every sampled user",
        "task": "split 100 points across 5 archetypes + confidence (High/Medium/Low)",
        "annotators": args.annotators,
        "n_users": len(cards),
        "per_class": args.per_class,
        "seed": args.seed,
        "blinded": True,
        "source": "247 real Opus-labeled users (slack/nankani/mbada); synthetic excluded",
        "parity": f"last {MAX_MESSAGES_PER_USER} msgs, [I]/[R] tagged, body<= {BODY_TRUNC} chars — identical to opus input",
    }
    with open(config.ANNOTATION_ASSIGNMENT, "w", encoding="utf-8") as f:
        json.dump(assignment, f, ensure_ascii=False, indent=2)

    # ── per-annotator app (independent shuffle of card order) ──────────────────
    for a in args.annotators:
        order = list(cards)
        random.Random(args.seed + hash(str(a)) % 10000).shuffle(order)
        folder = os.path.join(config.ANNOTATION, f"annotator_{a}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "label_app.html"), "w", encoding="utf-8") as f:
            f.write(render_app(a, order, cheatsheet, REMINDERS))
        write_readme(folder, a, len(cards))
        print(f"  → app for annotator '{a}': {folder}/label_app.html")

    os.makedirs(config.ANNOTATION_RETURNED, exist_ok=True)

    print(f"\n✅ Done.")
    print(f"   Shared (blinded) sample : {config.ANNOTATION_SAMPLE}")
    print(f"   Hidden answer key       : {config.ANNOTATION_ANSWER_KEY}  (DO NOT share)")
    print(f"   Give each teammate their folder: outputs/annotation/annotator_<name>/")
    print(f"   When CSVs come back, drop them in: {config.ANNOTATION_RETURNED}/  then run analyze_agreement.py")


if __name__ == "__main__":
    main()
