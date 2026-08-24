"""
Candidate Ranking Module

Scores candidate contexts based on:
- Exact lexical match
- Same entity
- Same hierarchy
- Relationship match
- Perturbation encounter
- 3×3 encounter
- Elastic-cloud encounter
- Freshness
- Weak lexical match (lower weight)
- Distant relationship
- Cloud encounter
- Unrelated hierarchy (penalty)

Weights are configured so strong matches dominate.
"""

from typing import List, Dict, Any, Optional
from tokenizer import tokenize, normalize_lang, letter_score, word_score
from grid_crawler import crawl as grid_crawl
from memory_grid import MemoryGrid
from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive
from gsp import calculate_lsum, calculate_ssum, first_letter_index, gsp_place, elastic_cloud

# Scoring weights
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
}

def compute_gsp_cell(text: str, lang: str = "en"):
    """Compute GSP cell for a text using Lsum, Ssum, first letter."""
    clean = normalize_lang(text)
    L = calculate_lsum(clean, lang)
    S = calculate_ssum(clean, lang)
    c = first_letter_index(clean, lang)
    placement = gsp_place(L, S, c, K=1, D=1, C=26, R=64)
    return placement["primary_cell"]  # dict with col,row

def compute_elastic_cloud(text: str, lang: str = "en"):
    """Compute elastic cloud cells for a text."""
    clean = normalize_lang(text, lang)
    L = calculate_lsum(clean, lang)
    S = calculate_ssum(clean, lang)
    c = first_letter_index(clean, lang)
    return elastic_cloud(L, S, c, radius=1, first_letter_radius=1, C=26, R=64)

def get_three_by_three(cell: Dict[str, int]) -> List[Dict[str, int]]:
    """Return 3×3 neighbor cells around the given cell."""
    col, row = cell["col"], cell["row"]
    cells = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cells.append({
                "col": (col + dc) % 26,
                "row": ((row - 1 + dr) % 64) + 1
            })
    return cells

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
    Score a candidate against a query using the rubric.
    Returns total score and breakdown.
    """
    scores = {key: 0.0 for key in WEIGHTS}

    # Tokenize and compute lexical scores
    q_tokens = tokenize(query)
    c_tokens = tokenize(candidate_text)

    # Exact lexical match: use word_score from tokenizer (exact/stem)
    exact_lex = word_score(q_tokens, candidate_text)
    scores["exact_lexical"] = min(exact_lex, 100) * WEIGHTS["exact_lexical"] / 100

    # Same entity: if candidate contains any query entity
    if query_entities and candidate_entities:
        if any(e in candidate_text.lower() for e in query_entities):
            scores["same_entity"] = WEIGHTS["same_entity"]

    # Same hierarchy: if both hierarchy values match
    if query_hierarchy and candidate_hierarchy:
        if query_hierarchy == candidate_hierarchy:
            scores["same_hierarchy"] = WEIGHTS["same_hierarchy"]

    # Relationship match: based on GSP cells distance
    q_cell = compute_gsp_cell(query)
    c_cell = compute_gsp_cell(candidate_text)
    if q_cell and c_cell:
        row_dist = min(abs(q_cell["row"] - c_cell["row"]), 64 - abs(q_cell["row"] - c_cell["row"]))
        col_dist = min(abs(q_cell["col"] - c_cell["col"]), 26 - abs(q_cell["col"] - c_cell["col"]))
        total_dist = row_dist + col_dist
        # Relationship score: closer = higher
        scores["relationship"] = max(0, WEIGHTS["relationship"] * (1 - total_dist / (64 + 26)))

    # Perturbation encounter: if candidate cell is in perturbation of query cell
    q_cell = compute_gsp_cell(query)
    c_cell = compute_gsp_cell(candidate_text)
    if q_cell and c_cell:
        # simple perturbation: same column with row difference <=1
        if abs(q_cell["col"] - c_cell["col"]) <= 1 and abs(q_cell["row"] - c_cell["row"]) <= 1:
            scores["perturbation"] = WEIGHTS["perturbation"]

    # 3×3 encounter: if candidate cell in 3×3 around query cell
    q_cell = compute_gsp_cell(query)
    c_cell = compute_gsp_cell(candidate_text)
    if q_cell and c_cell:
        neighbors = get_three_by_three(q_cell)
        if any(n["col"] == c_cell["col"] and n["row"] == c_cell["row"] for n in neighbors):
            scores["three_by_three"] = WEIGHTS["three_by_three"]

    # Elastic cloud encounter
    q_cloud = compute_elastic_cloud(query)
    c_cell = compute_gsp_cell(candidate_text)
    if any(cell["col"] == c_cell["col"] and cell["row"] == c_cell["row"] for cell in q_cloud):
        scores["elastic_cloud"] = WEIGHTS["elastic_cloud"]

    # Freshness
    scores["freshness"] = min(freshness_score, 5.0) / 5.0 * WEIGHTS["freshness"]

    # Weak lexical match (if exact is low but some letter/word overlap)
    weak_lex = letter_score(q_tokens, candidate_text)
    if weak_lex < 20:
        scores["weak_lexical"] = weak_lex / 20 * WEIGHTS["weak_lexical"]

    # Distant relationship: if no strong relationship but not too far
    if scores["relationship"] < WEIGHTS["relationship"] * 0.3:
        scores["distant_relationship"] = WEIGHTS["distant_relationship"] * 0.5

    # Cloud encounter (similar to elastic cloud, but lower)
    if scores["elastic_cloud"] > 0:
        scores["cloud_encounter"] = WEIGHTS["cloud_encounter"] * 0.5

    # Unrelated hierarchy penalty
    if query_hierarchy and candidate_hierarchy and query_hierarchy != candidate_hierarchy:
        scores["unrelated_hierarchy"] = WEIGHTS["unrelated_hierarchy"]

    total = sum(scores.values())
    return {
        "total": total,
        "scores": scores,
    }