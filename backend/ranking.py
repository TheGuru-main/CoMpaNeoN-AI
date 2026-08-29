"""
Ranking Module

Ranks candidate contexts using the existing Lominii knowledge architecture.

Signals:
- Exact lexical match
- Same entity
- Same hierarchy / domain
- GSP relationship proximity
- Perturbation encounter
- 3×3 encounter
- Elastic-cloud encounter
- Freshness
- Weak lexical match
- Distant relationship
- Cloud encounter
- Word-pair relationship
- Next-word continuation
- Phrase continuity
- Unrelated hierarchy penalty

Important:
- Tokenizer/grid remains independent from main GSP placement.
- Domain detection remains external to this module.
- Dictionary/synonym understanding remains external to this module.
- No embeddings/vector search are introduced here.
"""

import re

from typing import List, Dict, Any, Optional

from tokenizer import (
    tokenize,
    normalize_lang,
    letter_score,
    word_score,
)

from keyboard import (
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    gsp_place,
    elastic_cloud,
)

from grid_crawler import crawl as grid_crawl
from memory_grid import MemoryGrid

from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive


# ---------------------------------------------------------------------------
# SCORING WEIGHTS
# ---------------------------------------------------------------------------

WEIGHTS = {
    "exact_lexical": 50,
    "same_entity": 50,
    "same_hierarchy": 50,
    "relationship": 30,
    "perturbation": 30,
    "three_by_three": 40,
    "elastic_cloud": 30,
    "freshness": 20,
    "weak_lexical": 10,
    "distant_relationship": 13,
    "cloud_encounter": 20,
    "unrelated_hierarchy": -20,
    "word_pair": 40,
    "next_word_prediction": 35,
    "phrase_continuity": 40,
}


# ---------------------------------------------------------------------------
# MAIN GSP CELL
# ---------------------------------------------------------------------------

def compute_gsp_cell(
    text: str,
    lang: str = "en",
) -> Optional[Dict[str, int]]:
    """
    Compute the main GSP placement cell.

    This is deliberately separate from tokenizer/grid placement.
    """

    if not text:
        return None

    lang = normalize_lang(lang)
    clean = str(text).strip().lower()

    if not clean:
        return None

    L = calculate_lsum(clean, lang)
    S = calculate_ssum(clean, lang)
    c = first_letter_index(clean, lang)

    placement = gsp_place(
        L,
        S,
        c,
        K=0,
        D=0,
        C=26,
        R=64,
    )

    return placement.get("primary_cell")


# ---------------------------------------------------------------------------
# ELASTIC CLOUD
# ---------------------------------------------------------------------------

def compute_elastic_cloud(
    text: str,
    lang: str = "en",
) -> List[Dict[str, int]]:
    """
    Compute elastic-cloud cells using the existing keyboard.py GSP logic.
    """

    if not text:
        return []

    lang = normalize_lang(lang)
    clean = str(text).strip().lower()

    if not clean:
        return []

    L = calculate_lsum(clean, lang)
    S = calculate_ssum(clean, lang)
    c = first_letter_index(clean, lang)

    return elastic_cloud(
        L,
        S,
        c,
        radius=1,
        first_letter_radius=1,
        C=26,
        R=64,
    )


# ---------------------------------------------------------------------------
# 3×3 ENCOUNTER
# ---------------------------------------------------------------------------

def get_three_by_three(
    cell: Dict[str, int],
) -> List[Dict[str, int]]:
    """
    Return surrounding cells around a GSP cell.
    """

    col = cell["col"]
    row = cell["row"]

    cells = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):

            if dr == 0 and dc == 0:
                continue

            cells.append({
                "col": (col + dc) % 26,
                "row": ((row - 1 + dr) % 64) + 1,
            })

    return cells


# ---------------------------------------------------------------------------
# WORD CLEANING
# ---------------------------------------------------------------------------

def _clean_words(text: str) -> List[str]:
    return re.findall(
        r"[a-zA-Z0-9_]+",
        str(text).lower(),
    )


# ---------------------------------------------------------------------------
# WORD PAIRS
# ---------------------------------------------------------------------------

def extract_word_pairs(
    text: str,
) -> List[str]:

    words = _clean_words(text)

    return [
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    ]


def word_pair_score(
    query: str,
    candidate_text: str,
) -> float:

    query_pairs = extract_word_pairs(query)

    if not query_pairs:
        return 0.0

    candidate_pairs = set(
        extract_word_pairs(candidate_text)
    )

    matched = sum(
        1
        for pair in query_pairs
        if pair in candidate_pairs
    )

    return (
        matched / len(query_pairs)
    ) * 100.0


# ---------------------------------------------------------------------------
# NEXT WORD
# ---------------------------------------------------------------------------

def next_word_prediction_score(
    query: str,
    candidate_text: str,
) -> float:

    query_words = _clean_words(query)
    candidate_words = _clean_words(candidate_text)

    if not query_words or not candidate_words:
        return 0.0

    q_len = len(query_words)

    if len(candidate_words) < q_len:
        return 0.0

    best_score = 0.0

    for i in range(
        len(candidate_words) - q_len + 1
    ):
        window = candidate_words[
            i:i + q_len
        ]

        matches = sum(
            1
            for q, c in zip(
                query_words,
                window,
            )
            if q == c
        )

        sequence_score = (
            matches / q_len
        )

        if sequence_score > best_score:
            best_score = sequence_score

    for i in range(
        len(candidate_words) - q_len
    ):
        window = candidate_words[
            i:i + q_len
        ]

        if window == query_words:
            return 100.0

    return best_score * 100.0


# ---------------------------------------------------------------------------
# PHRASE CONTINUITY
# ---------------------------------------------------------------------------

def phrase_continuity_score(
    query: str,
    candidate_text: str,
) -> float:

    query_words = _clean_words(query)
    candidate_words = _clean_words(candidate_text)

    if not query_words or not candidate_words:
        return 0.0

    longest = 0

    for i in range(len(query_words)):

        current = 0

        for j in range(len(candidate_words)):

            q_index = i + current

            if (
                q_index < len(query_words)
                and query_words[q_index]
                == candidate_words[j]
            ):
                current += 1
                longest = max(
                    longest,
                    current,
                )
            else:
                current = 0

    return (
        longest / len(query_words)
    ) * 100.0


# ---------------------------------------------------------------------------
# CANDIDATE RANKING
# ---------------------------------------------------------------------------

def score_candidate(
    query: str,
    candidate_text: str,
    candidate_entities: Optional[List[str]] = None,
    candidate_hierarchy: Optional[str] = None,
    query_entities: Optional[List[str]] = None,
    query_hierarchy: Optional[str] = None,
    freshness_score: float = 0.0,
    lang: str = "en",
) -> Dict[str, Any]:

    scores = {
        key: 0.0
        for key in WEIGHTS
    }

    if not query or not candidate_text:
        return {
            "total": 0.0,
            "scores": scores,
        }

    lang = normalize_lang(lang)

    q_tokens = tokenize(
        query,
        lang,
    )

    c_tokens = tokenize(
        candidate_text,
        lang,
    )

    if not q_tokens or not c_tokens:
        return {
            "total": 0.0,
            "scores": scores,
        }

    # -----------------------------------------------------------------------
    # WORD RELATIONSHIPS
    # -----------------------------------------------------------------------

    pair_score = word_pair_score(
        query,
        candidate_text,
    )

    scores["word_pair"] = (
        pair_score / 100.0
    ) * WEIGHTS["word_pair"]

    next_score = next_word_prediction_score(
        query,
        candidate_text,
    )

    scores["next_word_prediction"] = (
        next_score / 100.0
    ) * WEIGHTS["next_word_prediction"]

    phrase_score = phrase_continuity_score(
        query,
        candidate_text,
    )

    scores["phrase_continuity"] = (
        phrase_score / 100.0
    ) * WEIGHTS["phrase_continuity"]

    # -----------------------------------------------------------------------
    # EXACT LEXICAL
    # -----------------------------------------------------------------------

    exact_lex = word_score(
        q_tokens,
        candidate_text,
        lang,
    )

    scores["exact_lexical"] = (
        min(exact_lex, 100)
        * WEIGHTS["exact_lexical"]
        / 100
    )

    # -----------------------------------------------------------------------
    # ENTITY
    # -----------------------------------------------------------------------

    if query_entities and candidate_entities:

        q_entities = {
            str(e).lower()
            for e in query_entities
        }

        c_entities = {
            str(e).lower()
            for e in candidate_entities
        }

        if q_entities & c_entities:
            scores["same_entity"] = (
                WEIGHTS["same_entity"]
            )

    # -----------------------------------------------------------------------
    # HIERARCHY / DOMAIN
    # -----------------------------------------------------------------------

    if (
        query_hierarchy
        and candidate_hierarchy
    ):

        if query_hierarchy == candidate_hierarchy:

            scores["same_hierarchy"] = (
                WEIGHTS["same_hierarchy"]
            )

        else:

            scores["unrelated_hierarchy"] = (
                WEIGHTS["unrelated_hierarchy"]
            )

    # -----------------------------------------------------------------------
    # GSP RELATIONSHIP
    # -----------------------------------------------------------------------

    q_cell = compute_gsp_cell(
        query,
        lang,
    )

    c_cell = compute_gsp_cell(
        candidate_text,
        lang,
    )

    if q_cell and c_cell:

        row_distance = min(
            abs(
                q_cell["row"]
                - c_cell["row"]
            ),
            64 - abs(
                q_cell["row"]
                - c_cell["row"]
            ),
        )

        col_distance = min(
            abs(
                q_cell["col"]
                - c_cell["col"]
            ),
            26 - abs(
                q_cell["col"]
                - c_cell["col"]
            ),
        )

        total_distance = (
            row_distance
            + col_distance
        )

        scores["relationship"] = max(
            0.0,
            WEIGHTS["relationship"]
            * (
                1
                - total_distance / 90
            ),
        )

    # -----------------------------------------------------------------------
    # PERTURBATION
    # -----------------------------------------------------------------------

    if q_cell and c_cell:

        row_difference = abs(
            q_cell["row"]
            - c_cell["row"]
        )

        col_difference = abs(
            q_cell["col"]
            - c_cell["col"]
        )

        if (
            row_difference <= 1
            and col_difference <= 1
        ):
            scores["perturbation"] = (
                WEIGHTS["perturbation"]
            )

    # -----------------------------------------------------------------------
    # 3×3
    # -----------------------------------------------------------------------

    if q_cell and c_cell:

        neighbors = get_three_by_three(
            q_cell
        )

        if any(
            cell["col"] == c_cell["col"]
            and cell["row"] == c_cell["row"]
            for cell in neighbors
        ):
            scores["three_by_three"] = (
                WEIGHTS["three_by_three"]
            )

    # -----------------------------------------------------------------------
    # ELASTIC CLOUD
    # -----------------------------------------------------------------------

    q_cloud = compute_elastic_cloud(
        query,
        lang,
    )

    if c_cell:

        if any(
            cell["col"] == c_cell["col"]
            and cell["row"] == c_cell["row"]
            for cell in q_cloud
        ):
            scores["elastic_cloud"] = (
                WEIGHTS["elastic_cloud"]
            )

    # -----------------------------------------------------------------------
    # FRESHNESS
    # -----------------------------------------------------------------------

    scores["freshness"] = (
        min(
            max(freshness_score, 0.0),
            5.0,
        )
        / 5.0
    ) * WEIGHTS["freshness"]

    # -----------------------------------------------------------------------
    # WEAK LEXICAL
    # -----------------------------------------------------------------------

    weak_lex = letter_score(
        q_tokens,
        candidate_text,
        lang,
    )

    if weak_lex < 20:

        scores["weak_lexical"] = (
            weak_lex / 20
        ) * WEIGHTS["weak_lexical"]

    # -----------------------------------------------------------------------
    # DISTANT RELATIONSHIP
    # -----------------------------------------------------------------------

    if (
        scores["relationship"]
        < WEIGHTS["relationship"] * 0.3
    ):

        scores["distant_relationship"] = (
            WEIGHTS["distant_relationship"]
            * 0.5
        )

    # -----------------------------------------------------------------------
    # CLOUD ENCOUNTER
    # -----------------------------------------------------------------------

    if scores["elastic_cloud"] > 0:

        scores["cloud_encounter"] = (
            WEIGHTS["cloud_encounter"]
            * 0.5
        )

    # -----------------------------------------------------------------------
    # TOTAL
    # -----------------------------------------------------------------------

    total = sum(
        scores.values()
    )

    return {
        "total": total,
        "scores": scores,
    }
