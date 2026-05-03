"""
Step 4 — Deep-Dive Analysis (LLM call)
=======================================
Input  : state["scored_shortlist"]  (from Step 3)
         state["taste_profile"]     (from Step 1)
Output : state["deep_dives"]        — detailed analysis per top pick

For each top-scored item, the LLM writes a rich analysis that explains:
  - Why this specific title fits this specific user
  - What to expect tonally, thematically, and in terms of pacing
  - A personalised "watch/read if you liked X" hook
  - A fair warning about potential downsides for this user

This step cannot run without the scored shortlist from Step 3.
The analysis is personalised using the taste profile from Step 1.
"""

import json
from state import record_step, record_error
from grok_client import chat_json

DEEP_DIVE_COUNT = 3   # analyse the top N items in depth

SYSTEM_PROMPT = """You are a thoughtful cultural critic and personal recommendation advisor.
You will receive one candidate title, its match score/reasons, and the user's taste profile.

Write a personalised deep-dive analysis for this specific user.

Return ONLY valid JSON — no markdown fences, no text outside the JSON.

Output schema:
{
  "title": "<title>",
  "media_type": "<movie | book>",
  "one_line_pitch": "<compelling one-sentence pitch tailored to this user>",
  "why_this_user": "<2-3 sentences explaining specifically why this user will enjoy it>",
  "what_to_expect": {
    "tone": "<tone description>",
    "pacing": "<pacing description>",
    "themes_present": ["<theme1>", "<theme2>", ...]
  },
  "hook": "<'If you liked X, you'll love this because...' — use the user's example titles>",
  "fair_warning": "<honest note about any aspect this user might not enjoy>",
  "perfect_for_when": "<situational recommendation, e.g. 'a rainy weekend afternoon'>"
}"""

USER_PROMPT_TEMPLATE = """User's taste profile:
{profile}

Candidate to analyse:
Title      : {title}
Media type : {media_type}
Score      : {score}/10
Match reasons: {match_reasons}
Concerns   : {concerns}
Raw API data: {raw_data}

Write the deep-dive analysis now."""


def run(state: dict) -> dict:
    """
    Runs Step 4. Produces a deep-dive analysis for each top candidate.
    Mutates state in-place and returns it.
    """
    print("\n[Step 4] Writing deep-dive analyses for top picks...")

    profile = state["taste_profile"]
    shortlist = state["scored_shortlist"]

    if not shortlist:
        record_error(state, "step4_deep_dive", "Shortlist is empty; cannot deep-dive.")
        print("    [WARNING] No shortlist items to analyse.")
        state["deep_dives"] = []
        return state

    top_items = shortlist[:DEEP_DIVE_COUNT]
    deep_dives = []

    for i, item in enumerate(top_items):
        title = item.get("title", f"Item {i+1}")
        print(f"    Analysing [{i+1}/{len(top_items)}]: {title}...")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            profile=json.dumps(profile, indent=2),
            title=title,
            media_type=item.get("media_type", "?"),
            score=item.get("score", "?"),
            match_reasons=json.dumps(item.get("match_reasons", []), indent=2),
            concerns=json.dumps(item.get("concerns", []), indent=2),
            raw_data=json.dumps(item.get("raw_data", {}), indent=2),
        )

        try:
            dive = chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.4)
            dive["score"] = item.get("score")
            dive["concerns"] = item.get("concerns", [])
            deep_dives.append(dive)

        except Exception as exc:
            record_error(state, "step4_deep_dive", f"{title}: {exc}")
            print(f"    [ERROR] Deep-dive failed for '{title}': {exc}")
            deep_dives.append({
                "title": title,
                "media_type": item.get("media_type", "?"),
                "one_line_pitch": "Analysis unavailable.",
                "score": item.get("score"),
                "_error": str(exc),
            })

    state["deep_dives"] = deep_dives
    record_step(state, "step4_deep_dive")
    print(f"    Completed {len(deep_dives)} deep-dive analyses.")
    return state
