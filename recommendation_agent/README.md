# Movie / Book Recommendation Agent

A multi-step AI agent that accepts a natural-language preference description and produces
a personalised, structured recommendation report. Built from scratch with the Grok/Groq API
and free external data APIs — no LangChain or agent frameworks used.

---

## What It Does

The agent takes a free-text description of your preferences ("I love dark psychological
thrillers...") and runs it through a five-step chain where each step's output feeds directly
into the next:

| Step | Type      | Input from state                        | Output to state              |
|------|-----------|-----------------------------------------|------------------------------|
| 1    | LLM call  | `user_input`, `media_type`              | `taste_profile` (JSON)       |
| 2    | Tool call | `taste_profile`                         | `candidates` (API data)      |
| 3    | LLM call  | `taste_profile` + `candidates`          | `scored_shortlist`           |
| 4    | LLM call  | `taste_profile` + `scored_shortlist`    | `deep_dives`                 |
| 5    | LLM call  | `taste_profile` + `shortlist` + `deep_dives` | `final_report` (Markdown) |

No step can be removed without breaking the chain:
- Step 3 cannot run without both the taste profile (Step 1) and real API data (Step 2)
- Step 4 cannot run without the scored shortlist (Step 3)
- Step 5 cannot run without the deep-dives (Step 4) and full shortlist (Step 3)

The tool call in Step 2 retrieves real-world data from two free external APIs:
- **Open Library** (`openlibrary.org`) — for books (no API key required)
- **OMDb** (`omdbapi.com`) — for movies (free API key required)

---

## Chain Dependency Diagram

```
User input
    │
    ▼
[Step 1 — LLM]   →  taste_profile  {genres, themes, mood, era, example_titles, ...}
    │
    ▼
[Step 2 — TOOL]  →  candidates     ← real data: Open Library API + OMDb API
    │
    ▼
[Step 3 — LLM]   →  scored_shortlist   ← requires taste_profile + candidates
    │
    ▼
[Step 4 — LLM]   →  deep_dives         ← requires taste_profile + scored_shortlist
    │
    ▼
[Step 5 — LLM]   →  final_report       ← requires taste_profile + shortlist + deep_dives
    │
    ▼
outputs/report_<timestamp>.md
outputs/state_<timestamp>.json
```

---

## Project Structure

```
recommendation_agent/
├── agent.py                       # Main orchestrator — runs the 5-step chain
├── state.py                       # Shared state dict definition and helpers
├── grok_client.py                 # LLM API wrapper (Grok/Groq, all calls go here)
├── requirements.txt
├── README.md
├── .gitignore
├── steps/
│   ├── __init__.py
│   ├── step1_extract_profile.py   # LLM: raw input → structured taste profile
│   ├── step2_fetch_candidates.py  # TOOL: taste profile → API candidates
│   ├── step3_score_rank.py        # LLM: candidates + profile → scored shortlist
│   ├── step4_deep_dive.py         # LLM: top picks → personalised deep-dive analysis
│   └── step5_generate_report.py   # LLM: all state → final Markdown report
└── outputs/                       # Report and state files written here after each run
```

Each step module exposes a single `run(state: dict) -> dict` function.
The orchestrator in `agent.py` calls them in sequence, passing the same shared state dict.

---

## Installation

```bash
git clone <repo-url>
cd recommendation_agent
python3 -m pip install -r requirements.txt
```

---

## API Keys

### LLM API (required — choose one)

**Option A — Groq** (free tier, recommended for quick setup):
```bash
export GROQ_API_KEY=your_groq_key_here
```
Get a free key at: https://console.groq.com → API Keys → Create API Key

**Option B — Grok (xAI)**:
```bash
export GROK_API_KEY=your_grok_key_here
```
Get a key at: https://console.x.ai

If both are set, `GROQ_API_KEY` takes priority.

### OMDb API (optional — for movie recommendations)

```bash
export OMDB_API_KEY=your_omdb_key_here
```
Get a free key at: https://www.omdbapi.com/apikey.aspx

If not set, the agent runs in **book-only mode**. Open Library (books) requires no key.

---

## How to Run

### Interactive mode

```bash
python3 agent.py
```

You will be asked for:
1. Your preference description (free text)
2. Media type: `movie`, `book`, or `both`

### Non-interactive mode

```bash
python3 agent.py --input "I love slow-burn mysteries set in historical periods" --media book
```

### Arguments

| Argument  | Description                                | Default |
|-----------|--------------------------------------------|---------|
| `--input` | Preference description (skips the prompt)  | None    |
| `--media` | `movie`, `book`, or `both`                 | `both`  |

---

## What Inputs It Expects

Describe your preferences in plain English. Include any combination of:

- **Genres** — "psychological thrillers", "literary fiction", "sci-fi"
- **Themes** — "identity", "survival", "redemption", "friendship"
- **Mood** — "dark and tense", "feel-good", "thought-provoking", "light"
- **Example titles you loved** — "like Gone Girl and Shutter Island" (used to search APIs directly)
- **Things to avoid** — "no gore", "not too slow", "no romance"
- **Era** — "prefer modern", "love classics", "contemporary only"

Richer input gives better results, but minimal input ("I like thrillers") also works.

---

## Output

Two files are written to `outputs/` after each run:

| File                         | Contents                                          |
|------------------------------|---------------------------------------------------|
| `report_<timestamp>.md`      | Full structured Markdown recommendation report    |
| `state_<timestamp>.json`     | Complete agent state — all step inputs/outputs    |

The state JSON is especially useful for demos and inspection: it shows exactly what
each step received and produced, including errors.

---

## Error Handling and Graceful Degradation

The chain is designed to never crash hard:

- If the **tool call (Step 2)** fails or returns no results, Step 3 falls back to asking
  the LLM to generate recommendations from training knowledge, marking results with
  `"note": "LLM-generated (no API data available)"`.
- If any **LLM step** fails, the error is logged in `state["errors"]` and the chain
  continues with a safe default for that step's output.
- If Step 2 has no OMDb key, movies are skipped and only books are fetched.
- All errors are visible in `state["errors"]` in the output JSON.

---

## Sample Run

**Input:**
```
I like movies like 3 idiots, Chichore, and Swades — feel-good, inspirational,
friendship themes. Also lighthearted biography books in English.
```

**Chain output (terminal):**
```
[Step 1] Extracting taste profile from user input...
    Genres   : ['drama', 'biography', 'comedy']
    Themes   : ['friendship', 'inspiration', 'coming-of-age']
    Mood     : light
    Era      : any

[Step 2] Fetching candidates from external APIs (tool call)...
    Querying Open Library API for books...
    Retrieved 17 book candidates.
    Querying OMDb API for movies...
    Retrieved 12 movie candidates.

[Step 3] Scoring and ranking candidates against taste profile...
    Shortlisted 5 candidates:
      [9/10] 3 Idiots (movie)
      [8/10] Taare Zameen Par (movie)
      ...

[Step 4] Writing deep-dive analyses for top picks...
    Analysing [1/3]: 3 Idiots...
    ...

[Step 5] Generating final recommendation report...
    Report generated successfully.

Steps completed: step1_extract_profile, step2_fetch_candidates,
                 step3_score_rank, step4_deep_dive, step5_generate_report
```
