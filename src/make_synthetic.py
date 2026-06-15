"""
make_synthetic.py — disclosed synthetic augmentation for the rare archetypes.
═══════════════════════════════════════════════════════════════════════════════
WHY THIS EXISTS (read before trusting any number that uses it)
---------------------------------------------------------------------------
The real corpora (Slack / Nankani / Mbada) are technical Q&A and a professional
WhatsApp group; they are rich in Bee/Ant but structurally starved of Butterfly
(information-bridge) and Capybara (warm supporter). The other candidate real
source (Garimella Indian-WhatsApp) is broadcast political/religious/commercial
spam — *real* but catastrophically off-domain, so ingesting it would worsen the
train→team domain shift. See the README "Synthetic augmentation" section.

This module instead generates a small set of **naturalistic, in-domain student/
small-team project chats** in English (the language of our own team chat). The
conversations were written to read like real teammates, NOT as archetype
caricatures: nobody is "a Capybara", they are people coordinating a project, and
the archetype only emerges from how they happen to communicate. The labels are
assigned *afterwards*, by reading the chat exactly like any real user
(`data/labels/llm_labels.json`, dataset="synthetic").

VALIDITY GUARDRAILS
  • Disclosed: tagged dataset="synthetic" everywhere; never silently mixed in.
  • Ablation: train_ml can run with/without it (--no-synthetic); both reported.
  • The blind held-out TEAM test (wa_holdout_llm.json) stays 100 % real — the
    synthetic data never touches the honest scorecard.

Each team lives in its own time block weeks apart, so the 10-min (v1) and 45-min
(v2) reply windows never bridge two different teams in the single clean stream.
"""
import json
import config

# ── timing ────────────────────────────────────────────────────────────────────
# base epoch (ms) per team, spaced ~30 days apart so reply graphs never bridge.
BASE_MS   = 1_500_000_000_000
DAY_MS    = 86_400_000
TEAM_GAP  = 30 * DAY_MS
STEP_MS   = 3 * 60_000          # ~3 min between consecutive messages (varied below)


# Each team = ordered list of (author, body). Timestamps are synthesized so that
# authors interleave (→ real reply / mention structure). Authors are globally
# unique. Links and @mentions are included where a real teammate would use them.

TEAMS = [
    # ── Team 1 ── CS group project: build a small web app ─────────────────────
    # Priya=organizer/initiator · Daniel=heads-down builder · Sofia=resource bridge
    # Leila=encourager · Tom=low-engagement · Maya=helper/builder
    [
        ("Priya",  "Hey everyone 👋 kicking off our project channel. We have 3 weeks. Can we each say what part we want to own?"),
        ("Daniel", "I'll take the backend + database schema. Comfortable with Node/Postgres."),
        ("Sofia",  "I can do frontend. Also found this really clean React starter we could use: https://github.com/example/react-starter saves us a day of setup"),
        ("Leila",  "Amazing, thanks for finding that Sofia 🙌 this team is already moving fast"),
        ("Priya",  "Great. @Daniel can you sketch the schema by Wed so Sofia isn't blocked? I'll set up the repo + board tonight"),
        ("Daniel", "Yep, schema by Wed. I'll post an ERD."),
        ("Tom",    "ok sounds good"),
        ("Maya",   "I can pair with whoever gets stuck. Also happy to write the auth flow, I did something similar last sem"),
        ("Priya",  "Perfect @Maya take auth. Quick standup every Mon/Thu here, sound ok to everyone?"),
        ("Leila",  "Works for me! 😊"),
        ("Daniel", "Standups fine. I work better async but I'll show up."),
        ("Priya",  "Async updates totally count Daniel, just drop progress here. Repo is live, invites sent"),
        ("Sofia",  "Here's a Figma with a rough layout I mocked: https://figma.com/file/example feedback welcome, nothing is final"),
        ("Maya",   "This looks great Sofia. One thought — maybe move the nav to the top for mobile? Otherwise love it"),
        ("Leila",  "Agree, top nav is cleaner. You two are crushing it 💪"),
        ("Daniel", "ERD posted in the repo /docs/schema.png. Users, Projects, Tasks, plus a join table for assignments."),
        ("Priya",  "Nice and clean. @Sofia @Maya does that cover the frontend needs?"),
        ("Maya",   "Yep covers auth. Thanks Daniel 🙏"),
        ("Sofia",  "Works for me too. Starting on the components today"),
        ("Tom",    "sorry been swamped with another class, will catch up this weekend"),
        ("Priya",  "No worries Tom — can you take the README + deployment when you're back? Lower time pressure"),
        ("Tom",    "yeah ok"),
        ("Leila",  "We've got you Tom, just shout if you need anything 🙂"),
        ("Sofia",  "Sharing a good article on accessible forms, relevant to our signup page: https://web.dev/learn/forms"),
        ("Leila",  "Saving that, thank you! 🌟"),
        ("Daniel", "API stubs are up for projects + tasks. Endpoints documented in the README I started."),
        ("Maya",   "Auth is working on my branch, PR is up. @Daniel could you review the token logic when free?"),
        ("Daniel", "Reviewed, left 2 small comments, otherwise solid. Merging."),
        ("Maya",   "Thanks! Also @Sofia I wired the login form to the API, want to pair on the error states?"),
        ("Sofia",  "Yes please, 4pm? Also found a tiny toast library for the error messages: https://github.com/example/toast"),
        ("Maya",   "Perfect, 4pm works"),
        ("Priya",  "Mon standup: blockers only, keep it short 🙂 where are we vs the plan?"),
        ("Daniel", "Backend ~70%. Tasks CRUD done, assignments endpoint left."),
        ("Sofia",  "Frontend ~50%. Dashboard + login done, project view in progress"),
        ("Maya",   "Auth done, helping Sofia on the project view"),
        ("Leila",  "You're all so far ahead of where I expected, genuinely proud 🥹 I finished the test plan doc"),
        ("Tom",    "ok i pushed the readme draft and a deploy script, lmk if it works"),
        ("Priya",  "Tom the deploy script worked first try, nice 👏 @Daniel can you fill the env vars section?"),
        ("Daniel", "Added the env vars to the README."),
        ("Sofia",  "Heads up, found a CORS issue between front and back. Article that fixed it for me: https://web.dev/cross-origin"),
        ("Maya",   "Ah good catch, I hit that too. Fixed on my branch, pushing now"),
        ("Leila",  "Teamwork 💛 the demo is going to be so smooth"),
        ("Priya",  "Final stretch. @everyone please test the full flow tonight and log bugs on the board"),
        ("Daniel", "Tested backend paths, all green. One edge case with empty projects, fixing."),
        ("Maya",   "Found 2 small UI bugs, logged them. Can take both"),
        ("Sofia",  "I'll grab the styling polish ones"),
        ("Tom",    "tested deploy on staging, works"),
        ("Leila",  "Ran through it as a 'user' and it felt great. Tiny note: success message disappears too fast"),
        ("Maya",   "Good UX catch Leila, bumping the timeout"),
        ("Priya",  "We did it 🎉 demo tomorrow 10am. Thank you all, this was the best group I've had. Get some sleep 😴"),
        ("Leila",  "Couldn't have asked for better teammates 💛 see you all tomorrow"),
        ("Daniel", "👍 ship it"),
    ],

    # ── Team 2 ── Hackathon weekend team ──────────────────────────────────────
    # Arjun=rallier/initiator · Nina=resource bridge · Ben=builder
    # Grace=morale/supporter · Hana=design bridge · (small team, high intensity)
    [
        ("Arjun",  "ok team it's go time ⚡ 36 hours. Idea recap: campus carpool app. Everyone in?"),
        ("Nina",   "In! For maps I'd use the Mapbox free tier, docs here: https://docs.mapbox.com way easier than Google for this"),
        ("Ben",    "I'll own the backend + matching logic. Give me the data model and I'll start."),
        ("Grace",  "Let's gooo 🚀 I'll handle the pitch deck + demo script so we're not scrambling at hour 35"),
        ("Arjun",  "@Ben model: riders, drivers, routes, time windows. @Nina you + me on frontend map. Grace you're a lifesaver on the deck"),
        ("Hana",   "I'll float between frontend and design. Also dropping a Dribbble board for inspo: https://dribbble.com/tags/carpool"),
        ("Grace",  "These are gorgeous Hana 😍 ok everyone remember to eat and hydrate, we're in this together"),
        ("Ben",    "Schema up. riders, drivers, trips, matches. Starting the API."),
        ("Nina",   "Map is rendering with live markers. Sharing a CodeSandbox so you can all poke at it: https://codesandbox.io/s/example"),
        ("Arjun",  "love it. @Hana can you start on the match results card? @Grace how's the deck outline"),
        ("Grace",  "Deck outline done, just needs real screenshots. You're all moving so fast it's easy to keep up 💛"),
        ("Hana",   "Match card component done, also a quick style guide so we stay consistent: https://www.figma.com/example-styleguide"),
        ("Ben",    "Matching endpoint live. POST /match returns ranked drivers by detour cost."),
        ("Arjun",  "@Hana can you wire the match results into the UI? @Nina keep going on the map filters"),
        ("Hana",   "On it. Reusing your card component Nina, it's clean"),
        ("Nina",   "Go for it! Filters by time window + seats are working now"),
        ("Grace",  "Quick morale check 💛 you all are doing incredible. It's 2am, please drink water. We're 80% there"),
        ("Ben",    "Pushed a fix for the detour bug, was double-counting return trips."),
        ("Arjun",  "nice catch Ben. that was gonna bite us in the demo"),
        ("Hana",   "Match results are in the UI 🎉 looks real now"),
        ("Grace",  "AAA it looks amazing 🤩 taking screenshots now, thank you Hana"),
        ("Nina",   "Adding a geolocation 'find rides near me' button, found the API pattern here: https://developer.mozilla.org/geolocation"),
        ("Ben",    "Wired your geo button to the backend Nina, returns sorted by distance now"),
        ("Arjun",  "we are cooking 🔥 Judges in 2 hours. @Grace run us through the pitch once? @Ben deploy to the demo URL"),
        ("Ben",    "Deployed: demo url is in the pinned message. Stable. Seeded with 20 fake rides."),
        ("Grace",  "Pitch flow: problem → live demo → impact → ask. I'll open and close, Arjun you do the live demo. We've got this 🙌"),
        ("Hana",   "I made a one-pager handout for the judges too, sharing: https://www.figma.com/example-onepager"),
        ("Nina",   "Adding one more resource for the judges' Q on routing: the algorithm is basically this https://en.wikipedia.org/wiki/Vehicle_routing_problem"),
        ("Arjun",  "perfect, that'll sound solid. @Grace you good on time?"),
        ("Grace",  "Rehearsed twice, we're at 2:50, under the 3 min limit. You all are stars ⭐"),
        ("Hana",   "Did a final design pass, everything's aligned and on-brand now"),
        ("Ben",    "Health check is green, demo URL is stable. We're ready."),
        ("Arjun",  "Proud of this team regardless of result. Let's go show them 🔥"),
        ("Grace",  "Win or lose, loved every hour with you all 💛 go go go"),
        ("Nina",   "Let's do this 🚗💨"),
    ],

    # ── Team 3 ── Student club: charity event planning (warm group) ───────────
    # Fatima=warm leader · Liam=logistics/ant · Zara=resource bridge
    # Noah=supporter · Ella=supporter/cheerleader · Sam=low-engagement
    [
        ("Fatima", "Hi lovely people 🌸 so happy we're doing the charity bake sale together. First, thank you all for volunteering"),
        ("Liam",   "Happy to help. I'll handle logistics — tables, permits, the booking with facilities."),
        ("Zara",   "I'll do posters + social. Here's a free design tool that's great for this: https://www.canva.com and some past event photos for inspo"),
        ("Noah",   "This is going to be so good. Whatever needs doing, count me in 🙂"),
        ("Ella",   "Same! I'm not the most organized but I'm great at hyping people up 😄 thank you for leading Fatima"),
        ("Sam",    "ok i'm around"),
        ("Fatima", "Everyone's strengths are perfect for this. @Liam permits first since they take time. @Zara posters by Friday?"),
        ("Liam",   "Permit form submitted. Approval usually 3-4 days. Will chase if no reply by Thu."),
        ("Noah",   "You're so on it Liam 👏 let me know if you want a hand with the facilities people"),
        ("Zara",   "Posters drafted! Sharing the link, tell me honestly what to change: https://canva.com/design/example"),
        ("Noah",   "Honestly these are beautiful Zara, the colors pop 👏 maybe bigger date text?"),
        ("Ella",   "Agreed, just the date bigger! Otherwise stunning, you're so talented 💕"),
        ("Zara",   "Thank you both 🥰 updated, date is bigger now. Also found a cute font pairing guide: https://fontpair.co"),
        ("Fatima", "You all make this so easy. Sign-up sheet for baking slots is up — please add your name 💛"),
        ("Noah",   "Signed up for Saturday morning. Also I can bring my mom's brownies, they're famous 😄"),
        ("Ella",   "I'll do the lemonade stand! And I'll keep everyone's energy up on the day ☀️"),
        ("Liam",   "Tables confirmed, 6 of them, 9am setup. Permit approved this morning."),
        ("Ella",   "YESSS go Liam 🙌 we have a permit!!"),
        ("Fatima", "I'm genuinely grateful for this team 💛 @Sam would you like the cash box + change duty? It's important and low-stress"),
        ("Sam",    "sure i can do that"),
        ("Noah",   "Perfect role for you Sam, you're reliable 🙂"),
        ("Zara",   "Posting the event on the club page tonight, sharing the link so you can all reshare: https://instagram.com/p/example"),
        ("Noah",   "Reshared! Let's make some money for a good cause ❤️"),
        ("Ella",   "Reshared too and tagged a few friends 😄 the poster looks SO good on the feed Zara"),
        ("Fatima", "Quick checklist for Saturday: tables (Liam ✅), posters up (Zara), bakers confirmed (me), float (Sam), vibes (Ella 😄)"),
        ("Ella",   "Vibes will be IMMACULATE I promise ✨"),
        ("Liam",   "I'll arrive 8:30 to set up tables before everyone. Noah you said you could help carry?"),
        ("Noah",   "Yes! 8:30 works, I'll be there with the brownies and muscles 💪😄"),
        ("Zara",   "Sharing a playlist for the stall so it's not silent: https://open.spotify.com/playlist/example feel-good only"),
        ("Ella",   "Omg perfect, music makes everything better 🎶"),
        ("Fatima", "How's everyone feeling? Any worries before the day? No worry too small 💛"),
        ("Sam",    "all good on my end"),
        ("Noah",   "Feeling great honestly. This team makes volunteering actually fun"),
        ("Liam",   "Logistics all locked. We're good."),
        ("Fatima", "Setup at 9 — well, 8:30 for the early birds 😄 smiles on, and remember why we're doing this. Thank you, all of you 🌸"),
        ("Ella",   "So proud of us already 🙌 see you Saturday lovelies!"),
        ("Zara",   "Can't wait 🥰"),
    ],

    # ── Team 4 ── Exam study group ────────────────────────────────────────────
    # Kavya=organizer · Marco=resource bridge · Aisha=patient supporter/explainer
    # Jonas=builder/summarizer · Ravi=supporter · Mia=low-engagement
    [
        ("Kavya",  "Final is in 2 weeks 😬 should we set up a study schedule? I can organize topics by week"),
        ("Marco",  "Yes please. I have last year's past papers + a great YouTube playlist for the hard chapters: https://youtube.com/playlist?list=example"),
        ("Aisha",  "I can explain the dynamic programming stuff, I finally get it after struggling for weeks 😅 happy to walk anyone through it"),
        ("Jonas",  "I'll make a condensed summary sheet of all the formulas and share it."),
        ("Ravi",   "You all are so on top of this, thank you 🙏 I'll bring coffee to the sessions"),
        ("Mia",    "ok i'm in"),
        ("Kavya",  "Plan: Week 1 = ch 1-5, Week 2 = ch 6-9 + practice exam. @Aisha can you do a DP session Thursday?"),
        ("Aisha",  "Yes! Thursday 6pm. I'll prep examples and go slow, no question is too small, promise 🙂"),
        ("Ravi",   "Aisha you're a legend, DP has been haunting me 😅"),
        ("Marco",  "Dropping the past papers here, 2019-2023: https://drive.google.com/example sorted by topic"),
        ("Jonas",  "Formula sheet draft done — 2 pages, everything from the lectures: https://drive.google.com/example2"),
        ("Ravi",   "Jonas this is incredible, you saved us hours 🙌 thank you"),
        ("Aisha",  "This is so well organized Jonas 👏 I'll add little 'why' notes next to the tricky formulas if that helps people"),
        ("Jonas",  "That'd be great, go for it. Merged your notes into v2."),
        ("Kavya",  "@Mia anything specific you're stuck on? We can prioritize it"),
        ("Mia",    "graphs mostly"),
        ("Aisha",  "Let's add a graphs session then. @Mia I struggled with those too, we'll get through it together 💪"),
        ("Marco",  "Found a visualizer that makes graph algos click: https://visualgo.net/en super helpful for BFS/DFS"),
        ("Ravi",   "Saving that, thanks Marco. This group is the only reason I'm not panicking 😅"),
        ("Aisha",  "You're doing better than you think Ravi 🙂 panic is just caring a lot"),
        ("Kavya",  "DP session recap is in the doc for anyone who missed Thursday. Great turnout!"),
        ("Jonas",  "Updated the summary with the graph formulas Aisha mentioned."),
        ("Marco",  "Adding a Khan Academy refresher for probability, the chapter most people find dry: https://khanacademy.org/probability"),
        ("Ravi",   "Honestly bless this group chat 🙏 just did 3 past papers and actually felt okay"),
        ("Aisha",  "@Ravi want to go over the ones you missed? I'm free Sunday, totally judgment-free"),
        ("Ravi",   "That would genuinely help, thank you Aisha 🥹"),
        ("Kavya",  "Practice exam Saturday 10am, then we review answers together. Sound good?"),
        ("Aisha",  "Perfect. I'll explain anything anyone gets wrong, judgment-free zone 🙂"),
        ("Mia",    "thanks works for me"),
        ("Marco",  "I'll print copies of the practice exam so we can do it under real conditions"),
        ("Jonas",  "I'll time us and prep the answer key for the review after."),
        ("Ravi",   "Coffee's on me Saturday ☕ you all are the best, genuinely"),
        ("Kavya",  "Reminder: Saturday 10am, room 204. Bring nothing but a pen, we've got the rest 💪"),
        ("Aisha",  "You've all worked so hard. We're ready for this 🙂"),
        ("Mia",    "see you saturday"),
        ("Ravi",   "Win the exam, then we celebrate 🎉 thank you team"),
    ],

    # ── Team 5 ── Small side-project / startup team ───────────────────────────
    # Dev=founder/initiator · Lena=builder · Carlos=resource bridge
    # Yuki=morale/supporter · Ahmed=helper/supporter · Pia=low-engagement
    [
        ("Dev",    "Morning team ☀️ sprint planning. Goal this week: ship the onboarding flow. Who takes what?"),
        ("Lena",   "I'll build the onboarding screens + state management. Should have it by Thursday."),
        ("Carlos", "I'll handle copy + analytics. Also this Stripe guide is relevant for the paywall step later: https://stripe.com/docs/payments"),
        ("Yuki",   "Love the momentum 🎉 I'll QA each screen and keep our changelog updated so we celebrate the wins"),
        ("Ahmed",  "I'll take the email verification piece and help Lena with edge cases. Ping me anytime."),
        ("Pia",    "ok"),
        ("Dev",    "@Lena @Ahmed sync on the verification handoff? @Carlos analytics events list by Wed would unblock me"),
        ("Carlos", "Events list posted in Notion. Also sharing a good piece on onboarding best practices: https://www.lennysnewsletter.com/example"),
        ("Lena",   "Screens 1-3 done, PR up. Verification stub is ready for @Ahmed to fill in"),
        ("Ahmed",  "Got it. Filling in verification now, also added a friendlier error message for expired links 🙂"),
        ("Yuki",   "Just QA'd screens 1-3 — gorgeous work Lena 🙌 found one tiny typo, noted in the PR"),
        ("Lena",   "Fixed, thanks Yuki!"),
        ("Dev",    "moving fast, love it. @Carlos how's the copy reading?"),
        ("Carlos", "Copy draft in the doc. Kept it short + warm. Also a microcopy reference I love: https://www.nngroup.com/articles/microcopy"),
        ("Yuki",   "The copy made me smile honestly, especially the empty states 😄 nice work Carlos"),
        ("Ahmed",  "Verification done and tested, including the resend flow. @Lena it's wired into your screen 4."),
        ("Lena",   "Confirmed, screen 4 works end to end now. Thanks Ahmed 🙏"),
        ("Carlos", "Analytics live, we can now see drop-off per step. Sharing the dashboard link: https://example.analytics/dash"),
        ("Dev",    "This is exactly what we needed. Drop-off is at the verification step, interesting"),
        ("Ahmed",  "I bet it's the email delay. I can add a 'resend' nudge after 30s, quick fix"),
        ("Yuki",   "Great instinct Ahmed 👏 small touches like that are why our retention will be good"),
        ("Lena",   "Pushed the nudge UI for Ahmed's logic, looks clean"),
        ("Carlos", "Drop-off already down 8% since the nudge. Data's in the dashboard"),
        ("Dev",    "huge. @Yuki can you write up the win for the changelog?"),
        ("Yuki",   "WE SHIPPED ONBOARDING 🎉🎉 changelog updated, with a shout-out to everyone. So proud of this little team 💛"),
        ("Carlos", "Nice. One more resource for next sprint's retention work: https://example.com/retention-playbook"),
        ("Ahmed",  "Skimmed it, the cohort section is gold. Good find Carlos"),
        ("Dev",    "Great sprint. @Pia want to take a small piece next week to get ramped in? No pressure"),
        ("Pia",    "maybe yeah"),
        ("Ahmed",  "Happy to pair with you on it Pia whenever 🙂 we'll make it easy"),
        ("Yuki",   "You'll fit right in Pia 💛 this team is the kindest I've worked with"),
        ("Lena",   "Agreed. Retro: ship rate good, code quality good, vibes great. Let's keep it"),
        ("Carlos", "+1. Posting the retro notes + next sprint goals: https://notion.so/example-retro"),
        ("Dev",    "Thanks everyone. Small but mighty 💪 see you Monday"),
        ("Yuki",   "Retro vibe: we're small but we move and we're kind to each other. Love it 💛"),
        ("Pia",    "ok see you monday"),
    ],

    # ── Team 6 ── Student newsletter / content team (butterfly-heavy) ──────────
    # Iris/Theo/Dana=resource bridges · Nour=supporter · Felix=writer/ant · Oscar=low
    [
        ("Iris",   "Hey team! Putting together this week's issue. Dropping a few stories worth covering: https://www.theverge.com/example and https://arstechnica.com/example"),
        ("Theo",   "Nice finds. For the tools section I'd add this: https://www.producthunt.com/example and a great newsletter we could cross-link https://tldr.tech"),
        ("Nour",   "You two always surface the best stuff 🙌 honestly makes my job easy. Thank you"),
        ("Felix",  "I'll draft the lead article and edit submissions. Send me anything by Wednesday."),
        ("Dana",   "Curated 5 links for the 'around the web' box: https://example.com/links shortlisted from like 40, kept the strongest"),
        ("Oscar",  "ok"),
        ("Iris",   "@Felix here's the source for the lead: https://www.nature.com/example solid primary data, not a hot take"),
        ("Theo",   "Adding a design reference for the new header: https://www.behance.net/example clean and readable"),
        ("Nour",   "Felix your draft reads beautifully, the intro hooked me 🥰 small note added inline"),
        ("Felix",  "Thanks Nour, incorporated. Lead is final."),
        ("Dana",   "Found a better chart for the data story than what we had: https://ourworldindata.org/example clearer axis"),
        ("Iris",   "Perfect swap Dana. Also sharing a style guide for consistent headlines: https://www.apstylebook.com"),
        ("Theo",   "Cross-linked two partner newsletters, they'll reshare us. Links in the doc"),
        ("Nour",   "This is shaping up to be our best issue 💛 proud of everyone"),
        ("Felix",  "Final edit pass done. Ready to schedule."),
        ("Oscar",  "looks fine to me"),
        ("Iris",   "Scheduling for 8am. Thanks all — sharing the preview link: https://example.com/preview"),
        ("Dana",   "One more for next week's queue so we don't scramble: https://example.com/next 🙂"),
        ("Nour",   "You think ahead Dana, bless you 🙏"),
    ],

    # ── Team 7 ── Peer mentoring / wellness circle (capybara-heavy) ────────────
    # Amara/Bruno/Clara=supporters · Diego=resource bridge · Esra=organizer · Gabe=notes/ant
    [
        ("Esra",   "Hi everyone 🌿 setting up our weekly check-in. Tuesdays 7pm work? I'll send invites and keep us on track"),
        ("Amara",  "Works for me 💛 and just want to say — this circle has genuinely helped me this semester, thank you all"),
        ("Bruno",  "Same here. Whatever anyone's going through, I'm around to listen 🙂"),
        ("Clara",  "Tuesdays perfect. Sending everyone a little encouragement for midterms: you've got this, truly ✨"),
        ("Diego",  "If it helps, here's a free guided-breathing app a counselor recommended: https://www.headspace.com and a study-stress article https://example.edu/stress"),
        ("Gabe",   "I'll take notes each session and post action items so nothing gets lost."),
        ("Amara",  "Diego that app is wonderful, used it last night and slept better 🙏 thank you"),
        ("Bruno",  "Adding: anyone feeling overwhelmed this week, dm me, no judgment ever 💙"),
        ("Clara",  "Bruno you're the kindest. Reminder to everyone to drink water and be gentle with yourselves today ☀️"),
        ("Esra",   "Great check-in today. @Gabe could you post the action items?"),
        ("Gabe",   "Posted: 1) Amara tries the campus tutor, 2) Bruno shares his notes app, 3) group walk Friday."),
        ("Diego",  "For the group walk, found a nice quiet route: https://www.alltrails.com/example flat and pretty"),
        ("Amara",  "Friday walk sounds lovely 🌸 thank you for organizing us Esra"),
        ("Clara",  "Sending love to whoever needs it today 💕 you matter and you're doing better than you think"),
        ("Bruno",  "Beautifully said Clara. Proud of this little circle 🙂"),
        ("Esra",   "Next week same time. Take care of yourselves 🌿"),
        ("Gabe",   "Notes archived in the shared folder for anyone who missed it."),
        ("Clara",  "Thank you Gabe, always so reliable 💛"),
    ],

    # ── Team 8 ── Open-source docs sprint (butterfly + capybara) ──────────────
    # Hugo/Jad/Mona=resource bridges · Ines/Kira=supporters · Luca=low
    [
        ("Hugo",   "Kicking off the docs sprint 📚 here's the contribution guide and the style ref: https://developers.google.com/style and our issue board"),
        ("Ines",   "So happy to have new contributors this round! Welcome everyone, ask anything, we're a friendly bunch 🙂"),
        ("Jad",    "For the API examples I'd follow this pattern, it's the clearest I've seen: https://stripe.com/docs/api"),
        ("Kira",   "First-time contributors, you're already doing great by showing up 💛 ping me if setup is confusing"),
        ("Mona",   "Tagged a bunch of good-first-issues and linked references on each: https://github.com/example/issues makes onboarding smoother"),
        ("Luca",   "ok i'll grab one"),
        ("Hugo",   "Nice Luca! Here's the doc that explains that module: https://example.dev/module shout if stuck"),
        ("Ines",   "@Luca welcome aboard 🎉 honestly the first PR is the hardest, you've got this"),
        ("Jad",    "Added a markdown linting tool so our docs stay consistent: https://github.com/example/markdownlint"),
        ("Kira",   "Just reviewed two newcomer PRs — both lovely work, left only encouraging notes 🙂"),
        ("Mona",   "Sharing a great example of good docs for inspiration: https://docs.python.org tone is friendly but precise"),
        ("Hugo",   "Reference for the diagrams everyone asked about: https://mermaid.js.org renders in our markdown"),
        ("Ines",   "We merged 6 PRs today from 4 new people 🥳 so proud of this community"),
        ("Kira",   "Each of you made this better. Thank you for being patient and kind in reviews 💛"),
        ("Jad",    "Dropping a final checklist link so contributors can self-review: https://example.dev/checklist"),
        ("Luca",   "my pr is up"),
        ("Ines",   "@Luca yay!! 🎉 reviewing now, I'm sure it's great"),
        ("Mona",   "Curated next sprint's references already: https://example.dev/next so we hit the ground running"),
    ],

    # ── Team 9 ── Community moderators (capybara + butterfly) ─────────────────
    # Nils/Petra/Rosa=supporters · Otto/Quinn/Said=resource bridges
    [
        ("Otto",   "Monthly mod sync 🛡️ sharing the updated guidelines + a good moderation playbook: https://www.example.org/mod-playbook"),
        ("Nils",   "Thanks Otto. Reminder to everyone: lead with empathy, most 'rule-breakers' are just having a bad day 💙"),
        ("Petra",  "Welcoming 3 new mods this month! So glad you're here, we'll support you every step 🙂"),
        ("Quinn",  "For handling reports faster I set up these saved replies, feel free to use: https://example.org/canned-responses"),
        ("Rosa",   "New mods, you're going to do wonderfully 🌟 never hesitate to ask, no question is silly"),
        ("Said",   "Sharing a de-escalation guide that's helped me a lot: https://example.org/deescalation worth a read"),
        ("Nils",   "Said that guide is gold 🙏 the 'pause before replying' tip alone saved a thread yesterday"),
        ("Petra",  "Shout-out to Rosa for calmly handling that heated thread last week, textbook kindness 💛"),
        ("Rosa",   "Aw thank you Petra 🥰 we all carry each other"),
        ("Quinn",  "Added a dashboard link so we can see report volume and share load fairly: https://example.org/mod-dash"),
        ("Otto",   "Good call Quinn. Also updated the wiki with this month's policy notes: https://example.org/wiki"),
        ("Nils",   "Anyone feeling burnt out, please take a break, the team's got your back 💙 mod health first"),
        ("Said",   "Sharing a short article on mod burnout, it's real and worth naming: https://example.org/burnout"),
        ("Petra",  "Thank you for looking out for us Nils 🙂 this is the warmest mod team I've been on"),
        ("Rosa",   "Genuinely love this crew 💛 see you all next sync, take care"),
        ("Quinn",  "One more resource for the new folks — full tooling overview: https://example.org/tools 🙂"),
        ("Otto",   "Perfect, pinned it. Thanks everyone, great sync 🛡️"),
    ],
]


def build_stream():
    rows = []
    for ti, team in enumerate(TEAMS):
        t = BASE_MS + ti * TEAM_GAP
        for j, (author, body) in enumerate(team):
            rows.append({
                "author": author,
                "datetime": t,
                "body": body,
                "is_reply": False,           # recomputed downstream by features
                "parent_id": None,
                "source": "synthetic",
                "team": f"syn_team_{ti+1}",
            })
            # vary the gap a little so timing metrics aren't perfectly uniform
            t += STEP_MS + (j % 4) * 60_000
    return rows


if __name__ == "__main__":
    rows = build_stream()
    with open(config.SYNTHETIC_CLEAN, "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_users = len({r["author"] for r in rows})
    print(f"✅ {len(rows)} synthetic messages, {n_users} users across {len(TEAMS)} teams")
    print(f"   → {config.GARIMELLA_CLEAN.replace('garimella','synthetic')}")
