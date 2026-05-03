"""
Step 3 — Score and Rank Candidates (LLM call)
=============================================
Input  : state["taste_profile"]   (from Step 1)
         state["candidates"]       (from Step 2, the tool call)
Output : state["scored_shortlist"] — top N candidates with match scores and reasons

The LLM receives the taste profile AND the raw API results.
It scores each candidate 0-10 against the profile and explains why.
This step cannot run without both Step 1 and Step 2 outputs.
"""

import json
from state import record_step, record_error
from grok_client import chat_json

TOP_N = 5   # how many to keep in the shortlist

SYSTEM_PROMPT = """You are an expert recommender system.
You will receive:
  1. A user's taste profile (structured JSON)
  2. A list of candidate movies or books retrieved from real APIs

Your task is to score each candidate against the taste profile and return a
ranked shortlist of the best matches.

Return ONLY valid JSON — no markdown fences, no text outside the JSON.

Output schema:
{
  "shortlist": [
    {
      "title": "<exact title from candidate>",
      "media_type": "<movie | book>",
      "score": <integer 0-10>,
      "match_reasons": ["<reason 1>", "<reason 2>", ...],
      "concerns": ["<potential mismatch 1>", ...],
      "candidate_index": <index of this item in the candidates list, 0-based>
    },
    ...
  ]
}

Scoring rules:
- 9-10: Exceptional match — hits multiple genres, themes, and mood
- 7-8 : Strong match — hits most criteria
- 5-6 : Decent match — hits some criteria but with notable gaps
- 3-4 : Weak match — only peripheral overlap
- 0-2 : Poor match — conflicts with stated preferences or disliked elements
- Penalise heavily for anything in disliked_elements
- Rank by score descending
- Include at most 5 items in shortlist
- Be honest about concerns; do not oversell"""

USER_PROMPT_TEMPLATE = """Taste profile:
{profile}

Candidates (retrieved from APIs):
{candidates}

Score and rank the candidates. Return the top {top_n} as JSON."""


def _format_candidate(idx: int, c: dict) -> str:
    """Format a candidate dict as a readable string for the LLM."""
    if c.get("media_type") == "book":
        return (
            f"[{idx}] BOOK: \"{c.get('title', '?')}\" by {c.get('author', '?')} "
            f"({c.get('year', '?')}) | "
            f"Subjects: {', '.join(c.get('subjects', [])[:5])} | "
            f"Rating: {c.get('rating', 'N/A')} ({c.get('rating_count', '?')} votes)"
        )
    else:
        return (
            f"[{idx}] MOVIE: \"{c.get('title', '?')}\" ({c.get('year', '?')}) "
            f"dir. {c.get('director', '?')} | "
            f"Genre: {c.get('genre', '?')} | "
            f"Plot: {c.get('plot', '?')} | "
            f"IMDb: {c.get('rating', 'N/A')} ({c.get('rating_count', '?')} votes)"
        )


def run(state: dict) -> dict:
    """
    Runs Step 3. Scores candidates against the taste profile.
    Mutates state in-place and returns it.
    """
    print("\n[Step 3] Scoring and ranking candidates against taste profile...")

    profile = state["taste_profile"]
    candidates = state["candidates"]

    if not candidates:
        print("    [WARNING] No candidates to score. Generating LLM-only fallback shortlist...")
        state["scored_shortlist"] = _fallback_shortlist(profile)
        record_step(state, "step3_score_rank")
        return state

    # Format candidates for the prompt
    formatted = "\n".join(
        _format_candidate(i, c) for i, c in enumerate(candidates)
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        profile=json.dumps(profile, indent=2),
        candidates=formatted,
        top_n=TOP_N,
    )

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt)
        shortlist = result.get("shortlist", [])

        # Attach original candidate data to each shortlist item
        for item in shortlist:
            idx = item.get("candidate_index")
            if idx is not None and 0 <= idx < len(candidates):
                item["raw_data"] = candidates[idx]
            else:
                item["raw_data"] = {}

        state["scored_shortlist"] = shortlist
        record_step(state, "step3_score_rank")

        print(f"    Shortlisted {len(shortlist)} candidates:")
        for item in shortlist:
            print(f"      [{item.get('score', '?')}/10] {item.get('title', '?')} ({item.get('media_type', '?')})")

    except Exception as exc:
        record_error(state, "step3_score_rank", str(exc))
        print(f"    [ERROR] Step 3 failed: {exc}")
        state["scored_shortlist"] = []

    return state


def _fallback_shortlist(profile: dict) -> list[dict]:
    """
    If no API candidates were retrieved, ask the LLM to generate recommendations
    purely from its training knowledge. This is the graceful degradation path.
    """
    fallback_system = """You are a recommendation expert.
The external API returned no results. Based solely on the user's taste profile,
suggest 5 well-known titles (movies or books as appropriate).
Return ONLY valid JSON in this schema:
{
  "shortlist": [
    {
      "title": "<title>",
      "media_type": "<movie | book>",
      "score": <7-9>,
      "match_reasons": ["<reason>"],
      "concerns": [],
      "candidate_index": -1,
      "raw_data": {},
      "note": "LLM-generated (no API data available)"
    }
  ]
}"""
    fallback_user = f"Taste profile:\n{json.dumps(profile, indent=2)}"
    try:
        result = chat_json(fallback_system, fallback_user)
        return result.get("shortlist", [])
    except Exception:
        return []
