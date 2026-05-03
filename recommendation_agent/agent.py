"""
Movie / Book Recommendation Agent
==================================
Main orchestrator. Runs the five-step chain in order, passing the shared
state object from step to step.

Chain:
  Step 1 (LLM)  — Extract taste profile from user input
  Step 2 (TOOL) — Fetch real candidates from Open Library / OMDb APIs
  Step 3 (LLM)  — Score and rank candidates against the taste profile
  Step 4 (LLM)  — Write deep-dive analyses for top picks
  Step 5 (LLM)  — Synthesise everything into a structured Markdown report

Usage:
  python agent.py                        # interactive CLI
  python agent.py --input "I love..."   # non-interactive
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path so all modules can import state / grok_client
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "steps"))

from state import create_state
import step1_extract_profile
import step2_fetch_candidates
import step3_score_rank
import step4_deep_dive
import step5_generate_report


OUTPUTS_DIR = Path(__file__).parent / "outputs"


def run_chain(user_input: str, media_type: str) -> dict:
    """
    Execute the full 5-step chain.
    Returns the final state dictionary.
    """
    print("\n" + "=" * 60)
    print("  RECOMMENDATION AGENT — starting chain")
    print("=" * 60)
    print(f"  Media type : {media_type}")
    print(f"  Input      : {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
    print("=" * 60)

    state = create_state(user_input, media_type)

    # ── Step 1: Extract taste profile ───────────────────────────────────────
    state = step1_extract_profile.run(state)

    # ── Step 2: Fetch candidates (tool call) ────────────────────────────────
    state = step2_fetch_candidates.run(state)

    # ── Step 3: Score and rank candidates ───────────────────────────────────
    state = step3_score_rank.run(state)

    # ── Step 4: Deep-dive analyses ──────────────────────────────────────────
    state = step4_deep_dive.run(state)

    # ── Step 5: Generate final report ───────────────────────────────────────
    state = step5_generate_report.run(state)

    return state


def save_outputs(state: dict) -> tuple[Path, Path]:
    """
    Save the final Markdown report and the full state JSON to the outputs folder.
    Returns (report_path, state_path).
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown report
    report_path = OUTPUTS_DIR / f"report_{timestamp}.md"
    report_path.write_text(state["final_report"] or "", encoding="utf-8")

    # Full state (for inspection / demo)
    state_path = OUTPUTS_DIR / f"state_{timestamp}.json"
    # Remove raw_data blobs to keep the JSON readable
    clean_state = {k: v for k, v in state.items() if k != "candidates"}
    state_path.write_text(
        json.dumps(clean_state, indent=2, default=str),
        encoding="utf-8",
    )

    return report_path, state_path


def interactive_prompt() -> tuple[str, str]:
    """Ask the user for their preferences interactively."""
    print("\n" + "=" * 60)
    print("  MOVIE / BOOK RECOMMENDATION AGENT")
    print("=" * 60)
    print("\nDescribe what kind of movies or books you enjoy.")
    print("Be as specific as you like — genres, themes, mood, examples.\n")

    user_input = input("Your preferences: ").strip()
    if not user_input:
        print("No input provided. Using a sample input for demonstration.")
        user_input = (
            "I love psychological thrillers with unreliable narrators. "
            "I enjoyed Gone Girl, Shutter Island, and The Girl with the Dragon Tattoo. "
            "I prefer dark, tense atmospheres and plots with unexpected twists. "
            "I dislike slow-burn romances and anything too gory just for shock value."
        )

    print("\nMedia type options: movie | book | both")
    media_type = input("Media type [both]: ").strip().lower() or "both"
    if media_type not in ("movie", "book", "both"):
        print(f"  Unrecognised type '{media_type}'. Defaulting to 'both'.")
        media_type = "both"

    return user_input, media_type


def print_summary(state: dict, report_path: Path, state_path: Path) -> None:
    print("\n" + "=" * 60)
    print("  CHAIN COMPLETE")
    print("=" * 60)
    print(f"  Steps completed : {', '.join(state['steps_completed'])}")
    if state["errors"]:
        print(f"  Non-fatal errors: {len(state['errors'])}")
        for e in state["errors"]:
            print(f"    [{e['step']}] {e['error']}")
    print(f"\n  Report saved to : {report_path}")
    print(f"  State saved to  : {state_path}")
    print("\n" + "─" * 60)
    print("  REPORT PREVIEW (first 40 lines)")
    print("─" * 60)
    if state["final_report"]:
        lines = state["final_report"].splitlines()
        for line in lines[:40]:
            print("  " + line)
        if len(lines) > 40:
            print(f"  ... ({len(lines) - 40} more lines in the full report)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Movie/Book Recommendation Agent using Grok API"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="User preference description (skips interactive prompt)")
    parser.add_argument("--media", type=str, default=None,
                        choices=["movie", "book", "both"],
                        help="Media type: movie | book | both")
    args = parser.parse_args()

    # Validate API key early — accept either Grok (xAI) or Groq
    if not os.environ.get("GROK_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        print("\n[ERROR] No LLM API key found. Set one of:")
        print("  export GROQ_API_KEY=your_key   (free tier: https://console.groq.com)")
        print("  export GROK_API_KEY=your_key   (xAI: https://console.x.ai)\n")
        sys.exit(1)

    if args.input:
        user_input = args.input
        media_type = args.media or "both"
    else:
        user_input, media_type = interactive_prompt()

    state = run_chain(user_input, media_type)
    report_path, state_path = save_outputs(state)
    print_summary(state, report_path, state_path)


if __name__ == "__main__":
    main()
