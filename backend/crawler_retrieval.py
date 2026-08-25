"""
Crawler Retrieval Layer

Retrieves candidates by entering the memory grid through EVERY
query token's own word-grid position.

This does not replace:
- tokenizer.py
- memory_grid.py
- grid_crawler.py
- ranking.py
- intent_analyzer.py
- directives.py
- external.py

It orchestrates them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tokenizer import tokenize, normalize_lang
from grid_crawler import crawl as grid_crawl
from ranking import score_candidate
from directives import detect_directive
from intent_analyzer import detect_domain


class CrawlerRetrieval:
    def __init__(self, memory_grid):
        self.memory = memory_grid

    def _token_position(self, token: Dict[str, Any]) -> tuple[int, int]:
        word = token["word"]

        L = word["L"]
        S = word["word_S"]
        col = word["col"]

        row = ((L + S - 1) % self.memory.rows) + 1
        col = col % self.memory.cols

        return row, col

    def retrieve(
        self,
        query: str,
        lang: str = "en",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        lang = normalize_lang(lang)

        tokens = tokenize(query, lang)
        if not tokens:
            return []

        directive = detect_directive(query)
        domain = detect_domain(query)

        # ---------------------------------------------------------
        # ENTER THE GRID THROUGH EVERY QUERY WORD
        # ---------------------------------------------------------

        positions = []

        seen_positions = set()

        for index, token in enumerate(tokens):
            row, col = self._token_position(token)

            position = (row, col)

            if position in seen_positions:
                continue

            seen_positions.add(position)

            positions.append({
                "token_index": index,
                "token": token,
                "row": row,
                "col": col,
            })

        # ---------------------------------------------------------
        # CRAWL EACH ENTRY POINT
        # ---------------------------------------------------------

        candidates: Dict[int, Dict[str, Any]] = {}

        per_entry_limit = max(1, limit // max(len(positions), 1))

        for position in positions:
            results = grid_crawl(
                self.memory,
                position["row"],
                position["col"],
                limit=per_entry_limit,
            )

            for result in results:
                doc_id = result.get("doc_id")

                if doc_id is None:
                    continue

                if doc_id not in candidates:
                    text = self.memory.get_doc(doc_id)

                    if not text:
                        continue

                    candidates[doc_id] = {
                        "doc_id": doc_id,
                        "text": text,
                        "encounters": [],
                    }

                candidates[doc_id]["encounters"].append({
                    "token_index": position["token_index"],
                    "token": position["token"]["original"],
                    "row": position["row"],
                    "col": position["col"],
                })

        # ---------------------------------------------------------
        # RANK ALL RETRIEVED DOCUMENTS
        # ---------------------------------------------------------

        ranked = []

        for candidate in candidates.values():
            result = score_candidate(
                query,
                candidate["text"],
                query_entities=[],
                query_hierarchy=domain,
                candidate_entities=[],
                candidate_hierarchy=domain,
                freshness_score=0.0,
                lang=lang,
            )

            ranked.append({
                "doc_id": candidate["doc_id"],
                "text": candidate["text"],
                "score": result["total"],
                "scores": result["scores"],
                "encounters": candidate["encounters"],
                "directive": directive,
                "domain": domain,
            })

        ranked.sort(
            key=lambda item: (
                item["score"],
                len(item["encounters"]),
            ),
            reverse=True,
        )

        return ranked[:limit]

    def retrieve_context(
        self,
        query: str,
        lang: str = "en",
        limit: int = 10,
        context_limit: int = 3,
    ) -> Dict[str, Any]:

        ranked = self.retrieve(
            query=query,
            lang=lang,
            limit=limit,
        )

        return {
            "query": query,
            "language": normalize_lang(lang),
            "directive": detect_directive(query),
            "domain": detect_domain(query),
            "results": ranked,
            "context": "\n".join(
                item["text"]
                for item in ranked[:context_limit]
            ),
        }