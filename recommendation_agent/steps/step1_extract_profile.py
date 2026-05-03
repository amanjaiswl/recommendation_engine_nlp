"""
Step 1 — Extract Taste Profile (LLM call)
=========================================
Input  : state["user_input"], state["media_type"]
Output : state["taste_profile"]  — a structured dict describing what the user likes

The LLM is asked to return JSON so that downstream steps can rely on
explicit fields rather than parsing free text.
"""

from state import record_step, record_error
from grok_client import chat_json

SYSTEM_PROMPT = """You are a precise preference-extraction assistant.
Your job is to read a user's description of what they enjoy in movies or books,
and convert it into a structured JSON profile.

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON.
The JSON must follow this exact schema:

{
  "genres": ["<genre1>", ...],
  "themes": ["<theme1>", ...],
  "mood": "<one of: light, dark, mixed, thrilling, contemplative, adventurous>",
  "preferred_era": "<one of: classic (pre-1980), modern (1980-2010), contemporary (2010+), any>",
  "disliked_elements": ["<element1>", ...],
  "example_titles": ["<title1>", ...],
  "media_type": "<movie | book | both>",
  "extra_notes": "<any nuance that did not fit the fields above>"
}

Rules:
- genres: canonical genre names only (e.g. "thriller", "sci-fi", "literary fiction")
- themes: abstract ideas the user cares about (e.g. "redemption", "identity", "survival")
- disliked_elements: things the user explicitly dislikes or wants to avoid
- example_titles: titles the user mentioned as examples they loved
- extra_notes: keep brief; if nothing, use ""
- If the user does not mention a field, use sensible defaults or empty lists
- Do not invent preferences the user did not express"""

USER_PROMPT_TEMPLATE = """User's preference description:
\"\"\"{user_input}\"\"\"

Media type requested: {media_type}

Extract the taste profile now."""


def run(state: dict) -> dict:
    """
    Runs Step 1. Mutates state in-place and returns it.
    """
    print("\n[Step 1] Extracting taste profile from user input...")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_input=state["user_input"],
        media_type=state["media_type"],
    )

    try:
        profile = chat_json(SYSTEM_PROMPT, user_prompt)

        # Normalise: ensure all expected keys exist
        profile.setdefault("genres", [])
        profile.setdefault("themes", [])
        profile.setdefault("mood", "any")
        profile.setdefault("preferred_era", "any")
        profile.setdefault("disliked_elements", [])
        profile.setdefault("example_titles", [])
        profile.setdefault("media_type", state["media_type"])
        profile.setdefault("extra_notes", "")

        state["taste_profile"] = profile
        record_step(state, "step1_extract_profile")
        print(f"    Genres   : {profile['genres']}")
        print(f"    Themes   : {profile['themes']}")
        print(f"    Mood     : {profile['mood']}")
        print(f"    Era      : {profile['preferred_era']}")

    except Exception as exc:
        record_error(state, "step1_extract_profile", str(exc))
        print(f"    [ERROR] Step 1 failed: {exc}")
        # Provide a minimal fallback so the chain can continue
        state["taste_profile"] = {
            "genres": [],
            "themes": [],
            "mood": "any",
            "preferred_era": "any",
            "disliked_elements": [],
            "example_titles": [],
            "media_type": state["media_type"],
            "extra_notes": state["user_input"],
        }

    return state
