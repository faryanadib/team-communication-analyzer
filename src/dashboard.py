"""
dashboard.py — SocialCompass · COIN Team Communication Analyzer
═══════════════════════════════════════════════════════════════
A friendly mirror of how a team communicates. Every member is placed on the
five COIN collaboration archetypes (Bee · Ant · Butterfly · Capybara · Leech),
with a short, plain-language read of what drives that placement.

Run:  streamlit run src/dashboard.py
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import config
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SocialCompass · Team Analyzer", page_icon="🧭", layout="wide")

# ── Global styling polish ────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1250px;}
  h1, h2, h3 {letter-spacing:-.01em;}
  div[data-testid="stMetric"] {
      background: #ffffff0a; border: 1px solid #ffffff14; border-radius: 14px;
      padding: 14px 16px;
  }
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
  .stTabs [data-baseweb="tab-list"] {gap: 4px;}
  .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 18px;}
  .arch-card {margin-bottom:12px; padding:14px 16px; border-radius:12px;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:11px; font-weight:700; color:#fff;}
</style>
""", unsafe_allow_html=True)

# ── Archetype metadata ───────────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    "🐝 Bee": "#F4C430", "🐜 Ant": "#C8772E", "🦋 Butterfly": "#9B59B6",
    "🦫 Capybara": "#2ECC71", "🔴 Leech": "#E74C3C",
}
ARCHETYPE_DESC = {
    "🐝 Bee": "Connects people and sparks discussion — starts threads, asks questions, links groups together.",
    "🐜 Ant": "The steady worker — focused on the task, reliable, gets things delivered.",
    "🦋 Butterfly": "Shares and bridges — passes on links, news and summaries that connect topics.",
    "🦫 Capybara": "Keeps the team warm — encouragement, thanks and a calming, supportive tone.",
    "🔴 Leech": "Quiet participant — reads more than writes; usually a sign to check in, not a verdict.",
}
ARCHETYPE_SIGNALS = {
    "🐝 Bee": "Starts conversations · asks questions · mentions teammates · rich vocabulary",
    "🐜 Ant": "Task-focused wording · steady replies · dependable follow-through",
    "🦋 Butterfly": "Shares links & resources · longer messages · reformulates ideas",
    "🦫 Capybara": "Supportive & polite words · positive tone · acknowledges others",
    "🔴 Leech": "Few replies · rarely starts threads · low overall activity",
}
FIT_COLS    = ["bee_fit", "ant_fit", "butterfly_fit", "capybara_fit", "leech_fit"]
FIT_LABELS  = ["🐝 Bee", "🐜 Ant", "🦋 Butterfly", "🦫 Capybara", "🔴 Leech"]
FIT_EMOJIS  = ["🐝", "🐜", "🦋", "🦫", "🔴"]
KEY = {"🐝 Bee": "bee", "🐜 Ant": "ant", "🦋 Butterfly": "butterfly",
       "🦫 Capybara": "capybara", "🔴 Leech": "leech"}


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_clean():
    df = pd.read_json(config.WHATSAPP_CLEAN, lines=True)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    return df

@st.cache_data
def load_results():
    res = pd.read_json(config.WA_RESULTS, lines=True)
    raw = load_clean()
    counts = raw.groupby("author").size().rename("msg_count")
    days = raw.groupby("author")["datetime"].apply(lambda s: s.dt.date.nunique()).rename("active_days")
    res = res.merge(counts, on="author", how="left").merge(days, on="author", how="left")
    res["msg_count"]   = res["msg_count"].fillna(0).astype(int)
    res["active_days"] = res["active_days"].fillna(0).astype(int)
    for fc, pc in zip(FIT_COLS, ["bee_pct", "ant_pct", "butterfly_pct", "capybara_pct", "leech_pct"]):
        if fc not in res.columns and pc in res.columns:
            res[fc] = res[pc]
    if "fit_top" not in res.columns:
        res["fit_top"] = res[FIT_COLS].max(axis=1)
    if "confidence" not in res.columns:
        res["confidence"] = res["fit_top"]
    return res

df = load_results()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧭 SocialCompass — Team Communication Analyzer")
st.caption("How your team collaborates, seen through the five COIN archetypes: "
           "🐝 Bee · 🐜 Ant · 🦋 Butterfly · 🦫 Capybara · 🔴 Leech.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Team members", len(df))
k2.metric("Messages analysed", int(df["msg_count"].sum()))
k3.metric("Roles present", df["archetype"].nunique())
k4.metric("Avg. confidence", f"{df['fit_top'].mean():.0f}%")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🧭 Overview", "🌐 Who talks to whom", "📅 Activity", "👤 Member profile"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1.35], gap="large")

    with col1:
        st.subheader("Team make-up")
        dist = df["archetype"].value_counts()
        colors = [ARCHETYPE_COLORS.get(a, "#999") for a in dist.index]
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        wedges, _t, autot = ax.pie(
            dist.values, labels=[a.split(" ", 1)[1] for a in dist.index],
            autopct="%1.0f%%", colors=colors, startangle=140, pctdistance=0.78,
            wedgeprops=dict(width=0.42, edgecolor="white"))
        for t in autot:
            t.set_color("white"); t.set_fontweight("bold")
        ax.set_title("Primary role of each member", fontsize=12, pad=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Who's who")
        for _, r in df.sort_values("fit_top", ascending=False).iterrows():
            a = r["archetype"]; c = ARCHETYPE_COLORS.get(a, "#999")
            conf = r.get("confidence", r["fit_top"])
            st.markdown(
                f"<div class='arch-card' style='border-left:5px solid {c}; background:{c}14'>"
                f"<span style='font-size:17px;font-weight:700'>{r['author']}</span> "
                f"<span style='font-size:15px'>{a}</span> "
                f"<span class='pill' style='background:{c}'>{conf:.0f}%</span><br>"
                f"<span style='font-size:12px;color:#9aa'>{int(r['msg_count'])} messages</span><br>"
                f"<span style='font-size:12px;color:#aab'>{ARCHETYPE_SIGNALS.get(a,'')}</span>"
                f"</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("How strongly each role fits")
    st.caption("Most people are a blend of roles. The bars show how much of each "
               "archetype every member shows — the tallest is their primary role.")

    fig, axes = plt.subplots(1, len(FIT_COLS), figsize=(14, 3.6), sharex=True)
    dfx = df.sort_values("fit_top", ascending=False)
    for i, (col, lab) in enumerate(zip(FIT_COLS, FIT_LABELS)):
        ax = axes[i]
        ax.barh(dfx["author"], dfx[col], color=ARCHETYPE_COLORS.get(lab, "#999"), alpha=.9)
        ax.set_xlim(0, max(60, df[FIT_COLS].max().max() + 8))
        ax.set_title(lab, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=.25)
        if i: ax.set_yticklabels([])
        for y, v in enumerate(dfx[col]):
            ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=8)
    plt.suptitle("Role strength per member (%)", y=1.03, fontsize=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("🏥 Team health")
    present = set(df["archetype"].unique())
    needed = {"🐝 Bee", "🐜 Ant", "🦋 Butterfly", "🦫 Capybara"}
    missing = needed - present
    if "🔴 Leech" in present:
        who = df[df["archetype"] == "🔴 Leech"]["author"].tolist()
        st.warning(f"⚠️ Low engagement flagged for **{', '.join(who)}** — a good moment "
                   "for a check-in, not a judgement of the person.")
    if missing:
        st.info(f"ℹ️ No-one currently leads as: {', '.join(missing)}. "
                "On a small, close team that's normal — see the role-strength bars above.")
    if not missing and "🔴 Leech" not in present:
        st.success("✅ Healthy mix — all four productive roles are covered and no-one is disengaged.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — INTERACTION NETWORK
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Who talks to whom")
    st.caption("Each circle is a member (bigger = more messages, colour = their role). "
               "Lines show replies within 10 minutes — thicker means they talk more often.")
    with st.spinner("Building the map…"):
        raw = load_clean().sort_values("datetime").reset_index(drop=True)
        members = set(df["author"].unique())
        G = nx.DiGraph()
        for _, r in df.iterrows():
            G.add_node(r["author"], archetype=r["archetype"])
        for i in range(1, len(raw)):
            a, prev = raw.iloc[i]["author"], raw.iloc[i - 1]["author"]
            gap = (raw.iloc[i]["datetime"] - raw.iloc[i - 1]["datetime"]).total_seconds() / 60
            if a != prev and a in members and prev in members and gap <= 10:
                if G.has_edge(a, prev): G[a][prev]["weight"] += 1
                else: G.add_edge(a, prev, weight=1)

    present = set(df["archetype"].unique())
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.set_facecolor("#0f1117"); fig.patch.set_facecolor("#0f1117")
    pos = nx.spring_layout(G, k=4, seed=42)
    sizes = [max(800, min(float(df[df["author"] == n]["msg_count"].iloc[0]) * 6, 5000))
             if n in members else 800 for n in G.nodes()]
    ncolors = [ARCHETYPE_COLORS.get(
        df[df["author"] == n]["archetype"].iloc[0] if n in members else "🔴 Leech", "#999")
        for n in G.nodes()]
    if G.edges():
        w = [G[u][v]["weight"] for u, v in G.edges()]; mw = max(w)
        nx.draw_networkx_edges(G, pos, ax=ax, width=[x / mw * 5 for x in w],
                               edge_color="#ffffff44", arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncolors, node_size=sizes, alpha=.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=10, font_weight="bold")
    ax.legend(handles=[mpatches.Patch(color=c, label=a.split(" ", 1)[1])
                       for a, c in ARCHETYPE_COLORS.items() if a in present],
              loc="lower left", facecolor="#1a1a2e", labelcolor="white", framealpha=.9)
    ax.axis("off"); plt.tight_layout(); st.pyplot(fig); plt.close()

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Conversations", G.number_of_edges())
    if G.number_of_nodes():
        ind, outd, deg = dict(G.in_degree()), dict(G.out_degree()), dict(G.degree())
        n2.metric("Most replied-to", max(ind, key=ind.get) if ind else "—")
        n3.metric("Replies the most", max(outd, key=outd.get) if outd else "—")
        n4.metric("Most connected", max(deg, key=deg.get) if deg else "—")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACTIVITY / TIMELINE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    raw = load_clean()
    raw["date"] = raw["datetime"].dt.date
    raw["hour"] = raw["datetime"].dt.hour
    daily = raw.groupby("date").size().reset_index(name="messages")

    st.subheader("Messages over time")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.fill_between(range(len(daily)), daily["messages"], alpha=.28, color="#F4C430")
    ax.plot(range(len(daily)), daily["messages"], color="#F4C430", lw=2, marker="o", ms=4)
    step = max(1, len(daily) // 18)
    ax.set_xticks(range(0, len(daily), step))
    ax.set_xticklabels([str(daily["date"].iloc[i]) for i in range(0, len(daily), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("messages"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("When is everyone online?")
    hourly = raw.groupby(["author", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(13, max(2.6, len(hourly) * .7)))
    im = ax.imshow(hourly.values, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.set_yticks(range(len(hourly))); ax.set_yticklabels(hourly.index.tolist(), fontsize=10)
    ax.set_xlabel("hour of day"); plt.colorbar(im, ax=ax, label="messages")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MEMBER PROFILE  (profile + the full role breakdown in one place)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    sel = st.selectbox("Choose a member",
                       df.sort_values("fit_top", ascending=False)["author"].tolist())
    if sel:
        u = df[df["author"] == sel].iloc[0]
        a = u["archetype"]; c = ARCHETYPE_COLORS.get(a, "#999")
        conf = u.get("confidence", u["fit_top"])
        probs = sorted(zip(FIT_LABELS, [float(u[col]) for col in FIT_COLS]),
                       key=lambda t: t[1], reverse=True)
        runner = f"{probs[1][0]} ({probs[1][1]:.0f}%)"
        gap = probs[0][1] - probs[1][1]
        strength = "a clear primary role" if gap >= 6 else "a balanced blend of two roles"
        st.markdown(
            f"<div style='padding:16px;border-left:6px solid {c};border-radius:10px;"
            f"background:{c}1c;margin-bottom:14px'>"
            f"<h2 style='margin:0'>{u['author']} &nbsp; {a}</h2>"
            f"<p style='margin:6px 0 0;color:#bcc;font-size:14px'>{ARCHETYPE_DESC.get(a,'')}</p>"
            f"<p style='margin:6px 0 0;color:#8a8;font-size:12px'>"
            f"Primary role {conf:.0f}% · next closest: {runner} · {strength}</p>"
            f"</div>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Messages", int(u["msg_count"]))
        m2.metric("Reply rate", f"{u['replies_sent']:.2f}")
        m3.metric("Active days", int(u["active_days"]))
        m4.metric("Avg length", f"{u['avg_msg_length']:.0f} w")
        m5.metric("Vocabulary", f"{u['MATTR']:.2f}")

        cR, cI = st.columns([1, 1], gap="large")
        with cR:
            st.markdown("#### Role profile")
            fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
            vals = [u[col] for col in FIT_COLS]
            ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
            vals2 = vals + vals[:1]; ang2 = ang + ang[:1]
            ax.plot(ang2, vals2, color=c, lw=2.4); ax.fill(ang2, vals2, color=c, alpha=.22)
            ax.set_xticks(ang); ax.set_xticklabels(FIT_EMOJIS, size=17)
            top = max(40, max(vals) + 8)
            ax.set_ylim(0, top); ax.set_yticks([15, 30]); ax.set_yticklabels(["15", "30"], size=8)
            ax.set_title(f"{u['author']} across the five roles", pad=18, size=11, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with cI:
            st.markdown("#### Role breakdown")
            fig, ax = plt.subplots(figsize=(5.0, 4.4))
            vals = [float(u[col]) for col in FIT_COLS]
            order = np.argsort(vals)
            ax.barh([FIT_LABELS[i] for i in order], [vals[i] for i in order],
                    color=[ARCHETYPE_COLORS[FIT_LABELS[i]] for i in order], alpha=.9)
            for y, i in enumerate(order):
                ax.text(vals[i] + .4, y, f"{vals[i]:.0f}%", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0, max(vals) + 8); ax.grid(axis="x", alpha=.25)
            ax.set_title("How strongly each role fits", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### What stands out")
        tips = []
        if u["msg_count"] == df["msg_count"].max(): tips.append("🏆 **Most active** member")
        if u["avg_msg_length"] == df["avg_msg_length"].max(): tips.append("📝 Writes the **longest** messages")
        if u["emotion_density"] == df["emotion_density"].max(): tips.append("😊 Most **emotionally expressive**")
        if u["MATTR"] == df["MATTR"].max(): tips.append("🐝 **Richest vocabulary**")
        if u["capybara_fit"] >= 22: tips.append("🦫 Strong **supportive** tendency")
        if u["bee_fit"] >= 24: tips.append("🐝 Strong **connector** tendency")
        if u["task_focus_score"] >= df["task_focus_score"].quantile(.75):
            tips.append("✅ High **task focus**")
        if u["question_ratio"] >= df["question_ratio"].quantile(.75):
            tips.append("❓ Asks the **most questions**")
        for t in tips or ["📊 A balanced contributor — no single signal dominates"]:
            st.markdown(f"- {t}")

        st.divider()
        st.markdown("#### Recent messages")
        msgs = load_clean()
        msgs = msgs[msgs["author"] == sel][["datetime", "body"]].tail(8).copy()
        msgs["datetime"] = msgs["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(msgs.reset_index(drop=True), use_container_width=True, hide_index=True)

st.divider()
st.caption(f"🧭 SocialCompass · {int(df['msg_count'].sum())} messages · {len(df)} members · "
           "five COIN collaboration archetypes")
