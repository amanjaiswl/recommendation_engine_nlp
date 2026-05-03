"""
Step 5 — Generate Structured Final Report (LLM call)
====================================================
Input  : state["taste_profile"]     (from Step 1)
         state["scored_shortlist"]  (from Step 3)
         state["deep_dives"]        (from Step 4)
Output : state["final_report"]      — a polished Markdown report string

The LLM synthesises everything accumulated in the state into a final,
human-readable recommendation report that a user can act on directly.

This step cannot run without deep_dives (Step 4) or scored_shortlist (Step 3).
"""

import json
from state import record_step, record_error
from grok_client import chat

SYSTEM_PROMPT = """You are a professional recommendation writer.
You will receive a user's taste profile, a scored shortlist of candidates, and
deep-dive analyses for the top picks.

Write a polished, well-structured Markdown recommendation report.

Structure the report EXACTLY as follows:

---
# Personalised Recommendation Report

## Your Taste Profile Summary
[2-3 sentence summary of what the user is looking for, in a warm, natural tone]

## Top Recommendations

### 1. [Title] ([Year]) — [media type]
**Match Score: X/10**
[one_line_pitch from deep dive]

**Why it's for you:**
[why_this_user from deep dive]

**What to expect:**
- Tone: [tone]
- Pacing: [pacing]
- Key themes: [themes_present]

**If you liked [example title]:** [hook]

**Fair warning:** [fair_warning]

**Perfect for:** [perfect_for_when]

---
[repeat for each top recommendation]

## Also Worth Considering
[List remaining shortlist items (not in top 3) as brief bullet points with score and one sentence]

## A Note on This Recommendation
[Brief honest note: how these were found (API + LLM reasoning), any caveats about data freshness]
---

Rules:
- Use the exact structure above
- Write in a warm, direct, second-person voice ("you'll love", "you might find")
- Do not invent information not present in the input
- Keep each recommendation section focused and readable — no rambling
- The "Also Worth Considering" section should have 1-3 items"""

USER_PROMPT_TEMPLATE = """Taste profile:
{profile}

Full shortlist (all scored items):
{shortlist}

Deep-dive analyses (top picks only):
{deep_dives}

Write the final recommendation report in Markdown now."""


def run(state: dict) -> dict:
    """
    Runs Step 5. Generates the final structured Markdown report.
    Mutates state in-place and returns it.
    """
    print("\n[Step 5] Generating final recommendation report...")

    profile = state["taste_profile"]
    shortlist = state["scored_shortlist"]
    deep_dives = state["deep_dives"]

    if not shortlist and not deep_dives:
        record_error(state, "step5_generate_report",
                     "No shortlist or deep-dives available.")
        state["final_report"] = "# Recommendation Report\n\nUnable to generate recommendations. Please try again with a more detailed preference description."
        return state

    # Summarise shortlist for the prompt (avoid huge token counts)
    shortlist_summary = []
    for item in shortlist:
        shortlist_summary.append({
            "title": item.get("title"),
            "media_type": item.get("media_type"),
            "score": item.get("score"),
            "match_reasons": item.get("match_reasons", []),
            "concerns": item.get("concerns", []),
        })

    user_prompt = USER_PROMPT_TEMPLATE.format(
        profile=json.dumps(profile, indent=2),
        shortlist=json.dumps(shortlist_summary, indent=2),
        deep_dives=json.dumps(deep_dives, indent=2),
    )

    try:
        report_md = chat(SYSTEM_PROMPT, user_prompt, temperature=0.5)
        state["final_report"] = report_md
        record_step(state, "step5_generate_report")
        print("    Report generated successfully.")

    except Exception as exc:
        record_error(state, "step5_generate_report", str(exc))
        print(f"    [ERROR] Step 5 failed: {exc}")
        state["final_report"] = _minimal_report(shortlist, deep_dives)

    return state


def _minimal_report(shortlist: list, deep_dives: list) -> str:
    """Fallback: produce a minimal report from raw data without LLM."""
    lines = ["# Recommendation Report\n",
             "_Note: Report generation partially failed. Showing raw results._\n"]
    for item in shortlist[:5]:
        lines.append(f"## {item.get('title', 'Unknown')} — {item.get('score', '?')}/10")
        for r in item.get("match_reasons", []):
            lines.append(f"- {r}")
        lines.append("")
    return "\n".join(lines)
