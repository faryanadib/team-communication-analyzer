"""
label_synthetic.py — LLM soft labels for the synthetic augmentation users.

Each label was assigned by READING the user's messages in synthetic_clean.json
(exactly like any real user), NOT by reading make_synthetic.py's persona
comments. Distributions are soft to reflect genuine overlap (a warm helper is
part Capybara, part Ant). All rows are confidence="medium" so the synthetic set
never outweighs the real labels (CONF_WEIGHT medium = 0.7).

Appending is idempotent: existing dataset="synthetic" rows are dropped first.
"""
import json
import config

# user_id -> soft label over (bee, ant, butterfly, capybara, leech) + note
SYNTH_LABELS = {
    # ── Team 1 ──
    "Priya":  dict(bee=.55, ant=.15, butterfly=.05, capybara=.15, leech=.10,
                   note="Initiates the channel, assigns tasks, sets standups, checks progress, heavy @mentions — classic connector/driver."),
    "Daniel": dict(bee=.10, ant=.62, butterfly=.05, capybara=.05, leech=.18,
                   note="Heads-down builder; terse substantive backend updates, prefers async, little affect."),
    "Sofia":  dict(bee=.15, ant=.20, butterfly=.50, capybara=.10, leech=.05,
                   note="Repeatedly surfaces tools/articles/Figma links for the team; also builds — information bridge."),
    "Leila":  dict(bee=.05, ant=.10, butterfly=.05, capybara=.60, leech=.20,
                   note="Almost purely encouragement/gratitude/praise + emojis; smooths and cheers, low task content."),
    "Tom":    dict(bee=.05, ant=.25, butterfly=.10, capybara=.05, leech=.55,
                   note="Low engagement, 'ok'/'busy'; eventually delivers README+deploy, so not pure leech."),
    "Maya":   dict(bee=.20, ant=.35, butterfly=.10, capybara=.30, leech=.05,
                   note="Offers to pair/help anyone stuck and ships auth — warm helper crossed with steady worker."),
    # ── Team 2 ──
    "Arjun":  dict(bee=.60, ant=.10, butterfly=.05, capybara=.15, leech=.10,
                   note="Rallies the team, assigns, coordinates the demo, @mentions everyone — driver."),
    "Nina":   dict(bee=.10, ant=.25, butterfly=.50, capybara=.10, leech=.05,
                   note="Brings the map docs, sandbox, geolocation + routing references — resource bridge who also codes."),
    "Ben":    dict(bee=.05, ant=.65, butterfly=.05, capybara=.05, leech=.20,
                   note="Backend owner; concise technical status, fixes bugs, deploys — steady worker."),
    "Grace":  dict(bee=.10, ant=.15, butterfly=.05, capybara=.60, leech=.10,
                   note="Morale officer: hydration/sleep reminders, praise, 'in this together'; also owns the deck."),
    "Hana":   dict(bee=.05, ant=.25, butterfly=.50, capybara=.15, leech=.05,
                   note="Design bridge: Dribbble board, style guide, one-pager; connects work into shareable artifacts."),
    # ── Team 3 ──
    "Fatima": dict(bee=.40, ant=.05, butterfly=.05, capybara=.40, leech=.10,
                   note="Warm leader — coordinates roles AND saturates the chat with gratitude/care; Bee/Capybara blend."),
    "Liam":   dict(bee=.10, ant=.65, butterfly=.05, capybara=.05, leech=.15,
                   note="Logistics: permits, tables, timing; reliable terse delivery — worker."),
    "Zara":   dict(bee=.05, ant=.20, butterfly=.55, capybara=.15, leech=.05,
                   note="Design + a stream of shared links (Canva, fonts, Instagram, Spotify) — information bridge."),
    "Noah":   dict(bee=.05, ant=.15, butterfly=.05, capybara=.60, leech=.15,
                   note="Supportive throughout, praises others, volunteers to help carry/bake — warm supporter."),
    "Ella":   dict(bee=.05, ant=.02, butterfly=.03, capybara=.70, leech=.20,
                   note="Self-described hype person; vibes/energy/emojis, minimal task content — strongest Capybara."),
    "Sam":    dict(bee=.025, ant=.20, butterfly=.025, capybara=.10, leech=.65,
                   note="Minimal one-word replies; accepts an assigned low-stress task — low-engagement participant."),
    # ── Team 4 ──
    "Kavya":  dict(bee=.55, ant=.20, butterfly=.05, capybara=.10, leech=.10,
                   note="Organizes the schedule and sessions, @mentions to unblock people — coordinator."),
    "Marco":  dict(bee=.05, ant=.20, butterfly=.60, capybara=.10, leech=.05,
                   note="Constantly shares papers, playlists, VisuAlgo, Khan links — quintessential resource bridge."),
    "Aisha":  dict(bee=.15, ant=.25, butterfly=.05, capybara=.45, leech=.10,
                   note="Patient explainer; offers 'judgment-free' help and reassurance — supporter/teacher with substance."),
    "Jonas":  dict(bee=.05, ant=.65, butterfly=.10, capybara=.10, leech=.10,
                   note="Produces the condensed formula/summary sheets; concise, output-focused — worker."),
    "Ravi":   dict(bee=.05, ant=.05, butterfly=.05, capybara=.65, leech=.20,
                   note="Gratitude, encouragement, brings coffee; low task content — warm supporter."),
    "Mia":    dict(bee=.025, ant=.20, butterfly=.025, capybara=.10, leech=.65,
                   note="Sparse, short ('graphs mostly', 'ok'); engages minimally — low-engagement."),
    # ── Team 5 ──
    "Dev":    dict(bee=.60, ant=.10, butterfly=.05, capybara=.10, leech=.15,
                   note="Runs sprint planning, assigns, @mentions, drives the goal — founder/driver."),
    "Lena":   dict(bee=.05, ant=.65, butterfly=.05, capybara=.10, leech=.15,
                   note="Ships onboarding screens, terse PR-driven updates — steady worker."),
    "Carlos": dict(bee=.05, ant=.25, butterfly=.55, capybara=.10, leech=.05,
                   note="Copy + analytics plus a steady stream of references (Stripe, NNG, playbooks) — bridge."),
    "Yuki":   dict(bee=.05, ant=.20, butterfly=.05, capybara=.60, leech=.10,
                   note="Morale + QA + celebratory changelogs; celebrates everyone's wins — supporter."),
    "Ahmed":  dict(bee=.10, ant=.38, butterfly=.07, capybara=.35, leech=.10,
                   note="'Ping me anytime', pairs with the newcomer, AND ships verification — helper/worker blend."),
    "Pia":    dict(bee=.025, ant=.15, butterfly=.025, capybara=.10, leech=.70,
                   note="Almost absent; 'ok'/'maybe yeah' — clearest low-engagement participant."),
    # ── Team 6 (newsletter, butterfly-heavy) ──
    "Iris":   dict(bee=.20, ant=.10, butterfly=.55, capybara=.10, leech=.05,
                   note="Drives the issue but mostly by surfacing stories/sources/style links — information bridge with light coordination."),
    "Theo":   dict(bee=.05, ant=.10, butterfly=.65, capybara=.15, leech=.05,
                   note="Constant stream of tools, design refs, cross-links — pure resource bridge."),
    "Nour":   dict(bee=.05, ant=.05, butterfly=.05, capybara=.70, leech=.15,
                   note="Almost entirely appreciation/encouragement of others' finds — warm supporter."),
    "Felix":  dict(bee=.05, ant=.68, butterfly=.05, capybara=.12, leech=.10,
                   note="Writes/edits the articles, concise and output-focused — steady worker."),
    "Dana":   dict(bee=.05, ant=.15, butterfly=.65, capybara=.10, leech=.05,
                   note="Curates and shortlists links, plans the queue ahead — bridge/curator."),
    "Oscar":  dict(bee=.025, ant=.15, butterfly=.05, capybara=.075, leech=.70,
                   note="One-word acks only ('ok', 'looks fine') — low-engagement."),
    # ── Team 7 (wellness circle, capybara-heavy) ──
    "Esra":   dict(bee=.55, ant=.20, butterfly=.10, capybara=.15, leech=.00,
                   note="Sets up the check-in, schedules, keeps the group on track — organizer/connector."),
    "Amara":  dict(bee=.05, ant=.05, butterfly=.05, capybara=.70, leech=.15,
                   note="Gratitude and warmth throughout, opens up and thanks people — supporter."),
    "Bruno":  dict(bee=.05, ant=.05, butterfly=.05, capybara=.70, leech=.15,
                   note="Offers to listen, 'dm me no judgment' — caring supporter."),
    "Clara":  dict(bee=.05, ant=.03, butterfly=.05, capybara=.72, leech=.15,
                   note="Pure encouragement/affirmations, low task content — strong Capybara."),
    "Diego":  dict(bee=.05, ant=.10, butterfly=.60, capybara=.20, leech=.05,
                   note="Shares apps, articles, a walking route — helpful resource bridge."),
    "Gabe":   dict(bee=.05, ant=.68, butterfly=.05, capybara=.12, leech=.10,
                   note="Reliable note-taker/action-items, archives sessions — quiet worker."),
    # ── Team 8 (docs sprint, butterfly + capybara) ──
    "Hugo":   dict(bee=.20, ant=.10, butterfly=.55, capybara=.10, leech=.05,
                   note="Opens the sprint and continuously points to guides/refs/modules — bridge with light lead."),
    "Ines":   dict(bee=.10, ant=.05, butterfly=.05, capybara=.65, leech=.15,
                   note="Welcomes and reassures newcomers throughout — warm supporter/host."),
    "Jad":    dict(bee=.05, ant=.15, butterfly=.65, capybara=.10, leech=.05,
                   note="Shares patterns, linting tools, checklists — resource bridge."),
    "Kira":   dict(bee=.05, ant=.10, butterfly=.05, capybara=.65, leech=.15,
                   note="Encourages first-timers, leaves kind review notes — supporter."),
    "Mona":   dict(bee=.05, ant=.10, butterfly=.70, capybara=.10, leech=.05,
                   note="Tags good-first-issues with references, curates next sprint — curator/bridge."),
    "Luca":   dict(bee=.025, ant=.25, butterfly=.05, capybara=.075, leech=.60,
                   note="Mostly silent, but does grab an issue and ship a PR — low-engagement, slight worker."),
    # ── Team 9 (moderators, capybara + butterfly) ──
    "Nils":   dict(bee=.10, ant=.05, butterfly=.10, capybara=.65, leech=.10,
                   note="Leads with empathy, watches for burnout, supports the team — warm supporter."),
    "Otto":   dict(bee=.25, ant=.10, butterfly=.50, capybara=.10, leech=.05,
                   note="Runs the sync and shares playbook/wiki/policy links — bridge with coordination."),
    "Petra":  dict(bee=.05, ant=.05, butterfly=.05, capybara=.70, leech=.15,
                   note="Welcomes new mods, gives shout-outs, gratitude — supporter."),
    "Quinn":  dict(bee=.05, ant=.15, butterfly=.65, capybara=.10, leech=.05,
                   note="Sets up saved replies, dashboards, tooling links — resource bridge."),
    "Rosa":   dict(bee=.05, ant=.05, butterfly=.05, capybara=.70, leech=.15,
                   note="Calm, kind, 'we carry each other' — strong Capybara."),
    "Said":   dict(bee=.05, ant=.10, butterfly=.65, capybara=.15, leech=.05,
                   note="Shares de-escalation and burnout guides — helpful resource bridge."),
}

ARCH = ["bee", "ant", "butterfly", "capybara", "leech"]


def main():
    # message counts from the clean stream
    from collections import Counter
    counts = Counter(json.loads(l)["author"]
                     for l in open(config.SYNTHETIC_CLEAN, encoding="utf-8"))

    rows = [json.loads(l) for l in open(config.LLM_LABELS, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("dataset") != "synthetic"]      # idempotent

    for user, lab in SYNTH_LABELS.items():
        s = sum(lab[a] for a in ARCH)
        assert abs(s - 1.0) < 1e-6, f"{user} sums to {s}"
        rows.append({
            "user_id": user, "dataset": "synthetic",
            **{a: lab[a] for a in ARCH},
            "confidence": "medium",
            "notes": lab["note"],
            "n_messages_total": counts[user],
            "n_messages_analyzed": counts[user],
        })

    with open(config.LLM_LABELS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_syn = sum(1 for r in rows if r.get("dataset") == "synthetic")
    print(f"✅ llm_labels.json now {len(rows)} labels (+{n_syn} synthetic)")
    from collections import Counter as C
    print("   synthetic argmax:",
          dict(C(max(ARCH, key=lambda a: l[a]) for l in SYNTH_LABELS.values())))


if __name__ == "__main__":
    main()
