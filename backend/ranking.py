"""
Candidate Ranking Module

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
- Main GSP placement continues to use the FULL WORD.
- Tokenizer/grid remains independent from main GSP placement.
- Domain detection remains external to this module.
- Dictionary/synonym understanding remains external to this module.
- No embeddings/vector search are introduced here.
"""

from typing import List, Dict, Any, Optional
import re

from tokenizer import (
    tokenize,
    normalize_lang,
    letter_score,
    word_score,
)

from grid_crawler import crawl as grid_crawl
from memory_grid import MemoryGrid

from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive

from gsp import (
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    gsp_place,
    elastic_cloud,
)


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

    "freshness": 10,
    "weak_lexical": 10,

    "distant_relationship": 10,
    "cloud_encounter": 10,

    "unrelated_hierarchy": -20,

    # Language relationship signals
    "word_pair": 20,
    "next_word_prediction": 25,
    "phrase_continuity": 20,
}


# ---------------------------------------------------------------------------
# BASIC TEXT HELPERS
# ---------------------------------------------------------------------------

def _clean_words(text: str) -> List[str]:
    """
    Convert text into normalized searchable words.

    Underscores are preserved intentionally.

    Example:

        where_can_I_buy_rice

    remains:

        ["where_can_I_buy_rice"]

    while ordinary text becomes:

        "where can I buy rice"
        →
        ["where", "can", "i", "buy", "rice"]
    """

    return re.findall(
        r"[a-zA-Z0-9_]+",
        str(text).lower(),
    )


# ---------------------------------------------------------------------------
# WORD PAIRS
# ---------------------------------------------------------------------------

def extract_word_pairs(text: str) -> List[str]:
    """
    Generate adjacent word relationships.

    Example:

        where can I buy rice

    becomes:

        where_can
        can_i
        i_buy
        buy_rice

    These are relationship signals only.

    They do NOT replace the main full-word GSP placement.
    """

    words = _clean_words(text)

    if len(words) < 2:
        return []

    return [
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    ]


def word_pair_score(
    query: str,
    candidate_text: str,
) -> float:
    """
    Score how many exact adjacent word relationships
    from the query are preserved by the candidate.

    Exact adjacent pairs receive the strongest relationship
    signal.

    Example:

        Query:
            where can I buy rice

        Candidate:
            where can I buy rice in Eneka

        Shared pairs:

            where_can
            can_i
            i_buy
            buy_rice

        Result:
            100
    """

    query_pairs = extract_word_pairs(query)

    if not query_pairs:
        return 0.0

    candidate_pairs = set(
        extract_word_pairs(candidate_text)
    )

    if not candidate_pairs:
        return 0.0

    matched = sum(
        1
        for pair in query_pairs
        if pair in candidate_pairs
    )

    return (
        matched / len(query_pairs)
    ) * 100.0


# ---------------------------------------------------------------------------
# NEXT-WORD CONTINUATION
# ---------------------------------------------------------------------------

def next_word_prediction_score(
    query: str,
    candidate_text: str,
) -> float:
    """
    Measure whether a candidate naturally continues the
    user's query.

    Example:

        Query:
            where can I buy

        Candidate:
            where can I buy rice

        Sequence:

            where → can → i → buy → rice
                                  ↑
                             next word

    An exact query followed by another candidate word receives
    the strongest continuation score.

    This function does not invent a vocabulary or synonym system.
    Dictionary relationships remain the responsibility of the
    dictionary / word-understanding layers.
    """

    query_words = _clean_words(query)
    candidate_words = _clean_words(candidate_text)

    if not query_words or not candidate_words:
        return 0.0

    q_len = len(query_words)

    # A candidate must have at least one word after the query
    # before it can provide next-word evidence.
    if len(candidate_words) <= q_len:
        return 0.0

    best_score = 0.0

    for i in range(len(candidate_words) - q_len):

        window = candidate_words[
            i:i + q_len
        ]

        matches = sum(
            1
            for q, c in zip(query_words, window)
            if q == c
        )

        sequence_ratio = (
            matches / q_len
        )

        # Exact query sequence followed by another word.
        if window == query_words:

            next_word = candidate_words[
                i + q_len
            ]

            if next_word:
                best_score = max(
                    best_score,
                    100.0,
                )

        else:
            # Partial sequential continuation.
            best_score = max(
                best_score,
                sequence_ratio * 60.0,
            )

    return best_score


# ---------------------------------------------------------------------------
# PHRASE CONTINUITY
# ---------------------------------------------------------------------------

def phrase_continuity_score(
    query: str,
    candidate_text: str,
) -> float:
    """
    Measure the longest contiguous sequence of query words
    appearing in the candidate.

    This rewards preservation of word order rather than merely
    counting independent keywords.

    Example:

        Query:
            where can I buy rice

        Candidate:
            you can buy rice from a shop

        Shared continuous phrase:

            can → buy → rice

    The score is proportional to the longest continuous sequence.
    """

    query_words = _clean_words(query)
    candidate_words = _clean_words(candidate_text)

    if not query_words or not candidate_words:
        return 0.0

    q_len = len(query_words)

    if q_len == 1:
        return (
            100.0
            if query_words[0] in candidate_words
            else 0.0
        )

    longest = 0

    for query_start in range(q_len):

        for candidate_start in range(
            len(candidate_words)
        ):

            length = 0

            while (
                query_start + length < q_len
                and
                candidate_start + length
                < len(candidate_words)
                and
                query_words[
                    query_start + length
                ]
                ==
                candidate_words[
                    candidate_start + length
                ]
            ):
                length += 1

            longest = max(
                longest,
                length,
            )

    return (
        longest / q_len
    ) * 100.0


# ---------------------------------------------------------------------------
# GSP CELL
# ---------------------------------------------------------------------------

def compute_gsp_cell(
    text: str,
    lang: str = "en",
):
    """
    Compute the main GSP cell using the FULL WORD/TEXT.

    This intentionally does not replace the tokenizer's grid.
    """

    # Preserve the existing architecture.
    clean = normalize_lang(text)

    L = calculate_lsum(
        clean,
        lang,
    )

    S = calculate_ssum(
        clean,
        lang,
    )

    c = first_letter_index(
        clean,
        lang,
    )

    placement = gsp_place(
        L,
        S,
        c,
        K=1,
        D=1,
        C=26,
        R=64,
    )

    return placement["primary_cell"]


# ---------------------------------------------------------------------------
# ELASTIC CLOUD
# ---------------------------------------------------------------------------

def compute_elastic_cloud(
    text: str,
    lang: str = "en",
):
    """
    Compute the elastic cloud around the main GSP placement.
    """

    clean = normalize_lang(text)

    L = calculate_lsum(
        clean,
        lang,
    )

    S = calculate_ssum(
        clean,
        lang,
    )

    c = first_letter_index(
        clean,
        lang,
    )

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
# 3 × 3 CELL
# ---------------------------------------------------------------------------

def get_three_by_three(
    cell: Dict[str, int],
) -> List[Dict[str, int]]:
    """
    Return the surrounding 3×3 cells.

    The center cell itself is excluded.
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
                "row": (
                    (row - 1 + dr) % 64
                ) + 1,
            })

    return cells


# ---------------------------------------------------------------------------
# MAIN CANDIDATE SCORER
# ---------------------------------------------------------------------------

def score_candidate(
    query: str,
    candidate_text: str,
    candidate_entities: Optional[List[str]] = None,
    candidate_hierarchy: Optional[str] = None,
    query_entities: Optional[List[str]] = None,
    query_hierarchy: Optional[str] = None,
    freshness_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Score a candidate against a query.

    Returns:

        {
            "total": float,
            "scores": {
                ...
            }
        }

    The individual signals remain visible so ranking can be
    inspected/debugged later.
    """

    # ---------------------------------------------------------
    # INITIALIZE SCORE BREAKDOWN
    # ---------------------------------------------------------

    scores = {
        key: 0.0
        for key in WEIGHTS
    }

    # ---------------------------------------------------------
    # TOKENIZATION
    # ---------------------------------------------------------

    q_tokens = tokenize(
        query
    )

    c_tokens = tokenize(
        candidate_text
    )

    if not q_tokens:
        return {
            "total": 0.0,
            "scores": scores,
        }

    # ---------------------------------------------------------
    # WORD PAIR RELATIONSHIP
    # ---------------------------------------------------------

    pair_score = word_pair_score(
        query,
        candidate_text,
    )

    scores["word_pair"] = (
        pair_score / 100.0
    ) * WEIGHTS["word_pair"]

    # ---------------------------------------------------------
    # NEXT WORD / SEQUENCE CONTINUATION
    # ---------------------------------------------------------

    next_word_score = next_word_prediction_score(
        query,
        candidate_text,
    )

    scores["next_word_prediction"] = (
        next_word_score / 100.0
    ) * WEIGHTS[
        "next_word_prediction"
    ]

    # ---------------------------------------------------------
    # PHRASE CONTINUITY
    # ---------------------------------------------------------

    phrase_score = phrase_continuity_score(
        query,
        candidate_text,
    )

    scores["phrase_continuity"] = (
        phrase_score / 100.0
    ) * WEIGHTS[
        "phrase_continuity"
    ]

    # ---------------------------------------------------------
    # EXACT LEXICAL MATCH
    # ---------------------------------------------------------

    exact_lex = word_score(
        q_tokens,
        candidate_text,
    )

    scores["exact_lexical"] = (
        min(exact_lex, 100)
        * WEIGHTS["exact_lexical"]
        / 100
    )

    # ---------------------------------------------------------
    # ENTITY MATCH
    # ---------------------------------------------------------

    if (
        query_entities
        and candidate_entities
    ):

        candidate_text_lower = (
            candidate_text.lower()
        )

        if any(
            str(entity).lower()
            in candidate_text_lower
            for entity in query_entities
        ):
            scores["same_entity"] = (
                WEIGHTS["same_entity"]
            )

    # ---------------------------------------------------------
    # HIERARCHY / DOMAIN MATCH
    # ---------------------------------------------------------

    if (
        query_hierarchy
        and candidate_hierarchy
    ):

        if (
            query_hierarchy
            == candidate_hierarchy
        ):
            scores["same_hierarchy"] = (
                WEIGHTS["same_hierarchy"]
            )

        else:
            scores["unrelated_hierarchy"] = (
                WEIGHTS[
                    "unrelated_hierarchy"
                ]
            )

    # ---------------------------------------------------------
    # MAIN GSP RELATIONSHIP
    # ---------------------------------------------------------

    q_cell = compute_gsp_cell(
        query
    )

    c_cell = compute_gsp_cell(
        candidate_text
    )

    if q_cell and c_cell:

        row_difference = abs(
            q_cell["row"]
            - c_cell["row"]
        )

        col_difference = abs(
            q_cell["col"]
            - c_cell["col"]
        )

        # Circular grid distance.
        row_dist = min(
            row_difference,
            64 - row_difference,
        )

        col_dist = min(
            col_difference,
            26 - col_difference,
        )

        total_dist = (
            row_dist
            + col_dist
        )

        max_dist = 64 + 26

        scores["relationship"] = max(
            0.0,
            WEIGHTS["relationship"]
            * (
                1
                - total_dist / max_dist
            ),
        )

    # ---------------------------------------------------------
    # PERTURBATION ENCOUNTER
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 3 × 3 ENCOUNTER
    # ---------------------------------------------------------

    if q_cell and c_cell:

        neighbors = get_three_by_three(
            q_cell
        )

        if any(
            n["col"] == c_cell["col"]
            and
            n["row"] == c_cell["row"]
            for n in neighbors
        ):
            scores["three_by_three"] = (
                WEIGHTS["three_by_three"]
            )

    # ---------------------------------------------------------
    # ELASTIC CLOUD ENCOUNTER
    # ---------------------------------------------------------

    if c_cell:

        q_cloud = compute_elastic_cloud(
            query
        )

        if any(
            cell["col"]
            == c_cell["col"]
            and
            cell["row"]
            == c_cell["row"]
            for cell in q_cloud
        ):
            scores["elastic_cloud"] = (
                WEIGHTS["elastic_cloud"]
            )

    # ---------------------------------------------------------
    # FRESHNESS
    # ---------------------------------------------------------

    scores["freshness"] = (
        min(
            max(freshness_score, 0.0),
            5.0,
        )
        / 5.0
    ) * WEIGHTS["freshness"]

    # ---------------------------------------------------------
    # WEAK LETTER / LEXICAL MATCH
    # ---------------------------------------------------------

    weak_lex = letter_score(
        q_tokens,
        candidate_text,
    )

    if weak_lex < 20:

        scores["weak_lexical"] = (
            weak_lex / 20.0
        ) * WEIGHTS[
            "weak_lexical"
        ]

    # ---------------------------------------------------------
    # DISTANT RELATIONSHIP
    # ---------------------------------------------------------

    if (
        scores["relationship"]
        <
        WEIGHTS["relationship"] * 0.3
    ):

        scores[
            "distant_relationship"
        ] = (
            WEIGHTS[
                "distant_relationship"
            ] * 0.5
        )

    # ---------------------------------------------------------
    # CLOUD ENCOUNTER
    # ---------------------------------------------------------

    if scores["elastic_cloud"] > 0:

        scores["cloud_encounter"] = (
            WEIGHTS[
                "cloud_encounter"
            ] * 0.5
        )

    # ---------------------------------------------------------
    # TOTAL
    # ---------------------------------------------------------

    total = sum(
        scores.values()
    )

    return {
        "total": total,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# OPTIONAL BATCH RANKING HELPER
# ---------------------------------------------------------------------------

def rank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    query_entities: Optional[List[str]] = None,
    query_hierarchy: Optional[str] = None,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Rank a list of candidate dictionaries.

    Expected candidate shape:

        {
            "text": "...",
            "entities": [...],
            "hierarchy": "...",
            "freshness": 0
        }

    Existing candidate fields are preserved.
    """

    ranked = []

    for candidate in candidates:

        text = candidate.get(
            "text",
            "",
        )

        if not text:
            continue

        result = score_candidate(
            query=query,
            candidate_text=text,
            candidate_entities=candidate.get(
                "entities"
            ),
            candidate_hierarchy=candidate.get(
                "hierarchy"
            ),
            query_entities=query_entities,
            query_hierarchy=query_hierarchy,
            freshness_score=candidate.get(
                "freshness",
                0.0,
            ),
        )

        ranked_candidate = dict(
            candidate
        )

        ranked_candidate[
            "score"
        ] = result["total"]

        ranked_candidate[
            "score_breakdown"
        ] = result["scores"]

        ranked.append(
            ranked_candidate
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked