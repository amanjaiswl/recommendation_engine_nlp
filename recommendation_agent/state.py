"""
Shared state object that flows through the entire agent chain.
Each step reads from and writes to this dictionary.
"""

from typing import Any


def create_state(user_input: str, media_type: str) -> dict[str, Any]:
    """Initialise the chain state with the user's raw input."""
    return {
        # ── inputs ──────────────────────────────────────────────────────────
        "user_input": user_input,          # raw text from user
        "media_type": media_type,          # "movie", "book", or "both"

        # ── step 1: taste profile ────────────────────────────────────────────
        "taste_profile": None,             # structured dict extracted by LLM

        # ── step 2: tool call results ────────────────────────────────────────
        "candidates": [],                  # list of raw candidates from API
        "tool_errors": [],                 # any errors encountered during fetch

        # ── step 3: scored shortlist ─────────────────────────────────────────
        "scored_shortlist": [],            # top N candidates with scores+reasons

        # ── step 4: deep-dive analyses ───────────────────────────────────────
        "deep_dives": [],                  # detailed write-up per top pick

        # ── step 5: final report ─────────────────────────────────────────────
        "final_report": None,              # markdown string, written to file

        # ── meta ─────────────────────────────────────────────────────────────
        "errors": [],                      # non-fatal errors accumulated
        "steps_completed": [],             # which steps finished successfully
    }


def record_step(state: dict, step_name: str) -> None:
    state["steps_completed"].append(step_name)


def record_error(state: dict, step_name: str, message: str) -> None:
    state["errors"].append({"step": step_name, "error": message})
