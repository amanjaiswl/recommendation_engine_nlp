"""
Step 2 — Fetch Candidates (TOOL CALL)
======================================
Input  : state["taste_profile"]
Output : state["candidates"]  — list of dicts from external APIs

This step is a TOOL CALL, not an LLM call.
It queries two free external APIs:
  - Open Library API  (https://openlibrary.org/search.json) for books
  - OMDb API          (https://www.omdbapi.com/)            for movies

No LLM is involved here. The APIs return real-world data that the LLM
cannot reliably fabricate, which is exactly why this step exists.
"""

import os
import time
import requests
from state import record_step, record_error

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")
OPEN_LIBRARY_BASE = "https://openlibrary.org/search.json"
OMDB_BASE = "https://www.omdbapi.com/"

# How many candidates to collect per query
BOOKS_PER_QUERY = 5
MOVIES_PER_QUERY = 5
MAX_QUERIES = 3          # number of genre/theme queries to run


# ── Open Library ──────────────────────────────────────────────────────────────

def fetch_books(genres: list[str], themes: list[str], era: str,
                example_titles: list[str] | None = None) -> list[dict]:
    """Query Open Library for books matching genres, themes and example titles."""
    candidates = []
    seen_keys = set()

    queries = _build_queries(genres, themes, max_queries=MAX_QUERIES)
    # Also search directly for any example titles the user mentioned
    for title in (example_titles or [])[:2]:
        queries.append(title)

    for q in queries:
        try:
            params = {
                "q": q,
                "fields": "key,title,author_name,first_publish_year,subject,ratings_average,number_of_ratings_count",
                "limit": BOOKS_PER_QUERY,
                "language": "eng",
            }
            if era != "any":
                year_range = _era_to_year_range(era)
                if year_range:
                    params["publish_year"] = year_range

            resp = requests.get(OPEN_LIBRARY_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for doc in data.get("docs", []):
                key = doc.get("key", "")
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                candidates.append({
                    "media_type": "book",
                    "title": doc.get("title", "Unknown"),
                    "author": ", ".join(doc.get("author_name", [])[:2]),
                    "year": doc.get("first_publish_year"),
                    "subjects": doc.get("subject", [])[:8],
                    "rating": doc.get("ratings_average"),
                    "rating_count": doc.get("number_of_ratings_count"),
                    "ol_key": key,
                    "source_query": q,
                })

            time.sleep(0.3)   # be polite to the free API

        except requests.RequestException as exc:
            # Non-fatal: log and continue with remaining queries
            candidates.append({"_error": str(exc), "source_query": q})

    return candidates


# ── OMDb (movies) ─────────────────────────────────────────────────────────────

def fetch_movies(genres: list[str], themes: list[str], era: str,
                 example_titles: list[str] | None = None) -> list[dict]:
    """Query OMDb for movies matching genres/themes and example titles."""
    if not OMDB_API_KEY:
        return [{
            "_error": (
                "OMDB_API_KEY not set. Get a free key at https://www.omdbapi.com/apikey.aspx "
                "and set it with:  export OMDB_API_KEY=your_key"
            )
        }]

    candidates = []
    seen_titles = set()

    # Build genre/theme queries plus direct title lookups for example titles
    queries = _build_queries(genres, themes, max_queries=MAX_QUERIES)

    # Direct title searches for example titles the user mentioned — these anchor
    # the search in the user's actual taste and pull similar titles from OMDb
    for title in (example_titles or [])[:3]:
        queries.append(title)

    for q in queries:
        try:
            params = {
                "apikey": OMDB_API_KEY,
                "s": q,
                "type": "movie",
            }
            resp = requests.get(OMDB_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("Response") != "True":
                continue

            for item in data.get("Search", [])[:MOVIES_PER_QUERY]:
                title = item.get("Title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Fetch full details
                detail = _omdb_detail(item.get("imdbID", ""))
                if detail:
                    candidates.append(detail)

            time.sleep(0.3)

        except requests.RequestException as exc:
            candidates.append({"_error": str(exc), "source_query": q})

    return candidates


def _omdb_detail(imdb_id: str) -> dict | None:
    """Fetch full movie details by IMDb ID."""
    if not imdb_id:
        return None
    try:
        resp = requests.get(
            OMDB_BASE,
            params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        if d.get("Response") != "True":
            return None
        return {
            "media_type": "movie",
            "title": d.get("Title", "Unknown"),
            "year": d.get("Year"),
            "genre": d.get("Genre", ""),
            "director": d.get("Director", ""),
            "actors": d.get("Actors", ""),
            "plot": d.get("Plot", ""),
            "rating": d.get("imdbRating"),
            "rating_count": d.get("imdbVotes"),
            "imdb_id": imdb_id,
            "runtime": d.get("Runtime", ""),
            "awards": d.get("Awards", ""),
        }
    except Exception:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_queries(genres: list[str], themes: list[str], max_queries: int) -> list[str]:
    """Build search query strings from genres and themes."""
    terms = genres[:2] + themes[:2]
    if not terms:
        return ["popular highly rated"]

    queries = []
    # First query: top genre + top theme
    if genres and themes:
        queries.append(f"{genres[0]} {themes[0]}")
    # Second query: second genre alone
    if len(genres) > 1:
        queries.append(genres[1])
    # Third query: remaining themes
    if len(themes) > 1:
        queries.append(themes[1])
    # Fallback
    if not queries:
        queries = [" ".join(terms[:3])]

    return queries[:max_queries]


def _era_to_year_range(era: str) -> str | None:
    mapping = {
        "classic (pre-1980)": "[* TO 1979]",
        "modern (1980-2010)": "[1980 TO 2010]",
        "contemporary (2010+)": "[2011 TO *]",
    }
    return mapping.get(era)


# ── main entry ────────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Runs Step 2. Queries APIs based on the taste profile from Step 1.
    Mutates state in-place and returns it.
    """
    print("\n[Step 2] Fetching candidates from external APIs (tool call)...")

    profile = state["taste_profile"]
    media_type = profile.get("media_type", state["media_type"])
    genres = profile.get("genres", [])
    themes = profile.get("themes", [])
    era = profile.get("preferred_era", "any")

    all_candidates = []
    errors = []

    if media_type in ("book", "both"):
        print("    Querying Open Library API for books...")
        example_titles = profile.get("example_titles", [])
        book_results = fetch_books(genres, themes, era, example_titles)
        book_candidates = [c for c in book_results if "_error" not in c]
        book_errors = [c for c in book_results if "_error" in c]
        all_candidates.extend(book_candidates)
        errors.extend(book_errors)
        print(f"    Retrieved {len(book_candidates)} book candidates.")
        if book_errors:
            print(f"    Book fetch errors: {len(book_errors)}")

    if media_type in ("movie", "both"):
        print("    Querying OMDb API for movies...")
        example_titles = profile.get("example_titles", [])
        movie_results = fetch_movies(genres, themes, era, example_titles)
        movie_candidates = [c for c in movie_results if "_error" not in c]
        movie_errors = [c for c in movie_results if "_error" in c]
        all_candidates.extend(movie_candidates)
        errors.extend(movie_errors)
        print(f"    Retrieved {len(movie_candidates)} movie candidates.")
        if movie_errors:
            print(f"    Movie fetch errors: {len(movie_errors)}")
            for e in movie_errors:
                print(f"      {e['_error']}")

    state["candidates"] = all_candidates
    state["tool_errors"] = errors

    if not all_candidates:
        record_error(state, "step2_fetch_candidates",
                     "No candidates retrieved from any API.")
        print("    [WARNING] No candidates found. Subsequent steps will use fallback.")
    else:
        record_step(state, "step2_fetch_candidates")

    return state
