from __future__ import annotations

import hashlib

from typing import (
    List,
    Dict,
    Any,
)

from tokenizer import (
    tokenize,
    normalize_lang,
    letter_score,
    word_score,
)

from memory_grid import MemoryGrid

from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive

from page_cache import PageCache
from memory_cache import MemoryCache

from ranking import score_candidate
from intent_analyzer import detect_domain


class WordUnderstanding:
    """
    Word Understanding / retrieval layer for CoMpaNeoN.

    The memory architecture has THREE independent entry routes:

        1. Letter Grid
           └── letter-token mapping

        2. Word Grid
           └── complete word-token mapping
               26 × 26

               L   = word/name length
               uID = L
               S   = ΣuID
               S   = L

        3. Full-Text Grid
           └── existing full-text placement/mapping

    These routes are NOT merged at storage time.

    They converge only during retrieval so that candidates can be
    collected, compared, scored and ranked.

    MemoryGrid remains authoritative for the actual storage-cell
    calculations.
    """

    def __init__(
        self,
        memory_grid: MemoryGrid,
    ):
        self.memory = memory_grid

        self.page_cache = PageCache()
        self.memory_cache = MemoryCache()

    # ==================================================================
    # HASH
    # ==================================================================

    def _hash(
        self,
        text: str,
    ) -> str:
        """
        Generate a deterministic hash for query/cache operations.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ==================================================================
    # LETTER ROUTE
    # ==================================================================

    def _retrieve_letter_route(
        self,
        token: dict,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve through the Letter Grid.

        The tokenizer supplies the letter-token indices.

        This route does NOT use the Word Grid's L/S calculation.
        It does NOT use the Full-Text Grid's placement calculation.
        """

        results: List[Dict[str, Any]] = []

        letters = token.get(
            "letter",
            [],
        )

        if not letters:
            return results

        seen = set()

        for letter_index in letters:

            entries = self.memory.get_letters_at(
                letter_index
            )

            for entry in entries:

                key = (
                    entry.get("doc_id"),
                    entry.get("original"),
                )

                if key in seen:
                    continue

                seen.add(key)

                result = dict(entry)

                result["retrieval_route"] = "letter"
                result["route_score"] = 1.0

                results.append(result)

        return results

    # ==================================================================
    # WORD ROUTE
    # ==================================================================

    def _retrieve_word_route(
        self,
        token: dict,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve through the 26 × 26 Word Grid.

        Word Grid definition:

            L   = name length
            uID = L
            S   = ΣuID
            S   = L

        Therefore:

            (L, S)

        is the word's Word Grid coordinate.

        The actual coordinate conversion/storage convention belongs
        to MemoryGrid. WordUnderstanding does not create a second
        Word Grid formula here.
        """

        word = token.get("word")

        if not word:
            return []

        # --------------------------------------------------------------
        # The tokenizer is authoritative for the word token.
        # --------------------------------------------------------------

        L = int(
            word.get("L", 0)
        )

        S = int(
            word.get("word_S", L)
        )

        if L <= 0:
            return []

        # --------------------------------------------------------------
        # Prefer MemoryGrid's authoritative Word Grid resolver.
        #
        # This keeps storage and retrieval mathematically identical.
        # --------------------------------------------------------------

        if hasattr(
            self.memory,
            "get_word_cell",
        ):
            row, col = self.memory.get_word_cell(
                L,
                S,
            )

            entries = self.memory.get_words_at(
                row,
                col,
            )

            results = []

            for entry in entries:

                result = dict(entry)

                result["retrieval_route"] = "word"
                result["route_score"] = 1.0

                results.append(result)

            return results

        # --------------------------------------------------------------
        # Compatibility fallback.
        #
        # If the older MemoryGrid has not yet exposed get_word_cell(),
        # use the explicit 26 × 26 (L,S) coordinate.
        #
        # The permanent implementation should keep the resolver in
        # MemoryGrid so storage and retrieval cannot diverge.
        # --------------------------------------------------------------

        word_grid_size = getattr(
            self.memory,
            "WORD_GRID_SIZE",
            26,
        )

        row = (
            (L - 1)
            % word_grid_size
        ) + 1

        col = (
            S
            % word_grid_size
        )

        entries = self.memory.get_words_at(
            row,
            col,
        )

        results = []

        for entry in entries:

            result = dict(entry)

            result["retrieval_route"] = "word"
            result["route_score"] = 1.0

            results.append(result)

        return results

    # ==================================================================
    # FULL-TEXT ROUTE
    # ==================================================================

    def _retrieve_fulltext_route(
        self,
        token: dict,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve through the independent Full-Text Grid.

        This route does NOT use the Letter Grid mapping.

        This route does NOT use the Word Grid coordinate.

        The established Full-Text placement remains authoritative.

        For the existing full-text mapping:

            L = word length
            S = word S
            c = word column/index

            start_row = ((L + S - 1) % rows) + 1
            start_col = c % cols
        """

        word = token.get("word")

        if not word:
            return []

        L = int(
            word.get("L", 0)
        )

        S = int(
            word.get("word_S", L)
        )

        c = int(
            word.get("col", 0)
        )

        if L <= 0:
            return []

        # --------------------------------------------------------------
        # Full-Text Grid placement.
        # --------------------------------------------------------------

        start_row = (
            (L + S - 1)
            % self.memory.rows
        ) + 1

        start_col = (
            c
            % self.memory.cols
        )

        entries = self.memory.get_tokens_at(
            start_row,
            start_col,
        )

        results = []

        for entry in entries[:limit]:

            result = dict(entry)

            result["retrieval_route"] = "fulltext"
            result["route_score"] = 1.0

            results.append(result)

        return results

    # ==================================================================
    # THREE-ROUTE RETRIEVAL
    # ==================================================================

    def _retrieve_three_routes(
        self,
        query: str,
        lang: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Enter the memory architecture through all three independent
        routes.

                         QUERY
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        LETTER GRID    WORD GRID    FULL-TEXT GRID
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                      CANDIDATES
                           │
                           ▼
                       RANKING

        A candidate reached through multiple routes is retained as
        one candidate while its route evidence is accumulated.
        """

        tokens = tokenize(
            query,
            lang,
        )

        if not tokens:
            return []

        # --------------------------------------------------------------
        # Candidate collection.
        #
        # Do NOT globally discard a result simply because another route
        # already found the same document.
        #
        # Instead, preserve route evidence.
        # --------------------------------------------------------------

        candidates: Dict[Any, Dict[str, Any]] = {}

        for token in tokens:

            # ==========================================================
            # ROUTE 1 — LETTER GRID
            # ==========================================================

            letter_results = (
                self._retrieve_letter_route(
                    token
                )
            )

            self._merge_route_results(
                candidates,
                letter_results,
                "letter",
            )

            # ==========================================================
            # ROUTE 2 — WORD GRID
            # ==========================================================

            word_results = (
                self._retrieve_word_route(
                    token
                )
            )

            self._merge_route_results(
                candidates,
                word_results,
                "word",
            )

            # ==========================================================
            # ROUTE 3 — FULL-TEXT GRID
            # ==========================================================

            fulltext_results = (
                self._retrieve_fulltext_route(
                    token,
                    limit,
                )
            )

            self._merge_route_results(
                candidates,
                fulltext_results,
                "fulltext",
            )

        return list(
            candidates.values()
        )

    # ==================================================================
    # ROUTE MERGING
    # ==================================================================

    def _merge_route_results(
        self,
        candidates: Dict[Any, Dict[str, Any]],
        results: List[Dict[str, Any]],
        route: str,
    ) -> None:
        """
        Merge retrieval evidence from one route into the unified
        candidate collection.

        Storage remains separate.

        Only the retrieval evidence is unified here.
        """

        for result in results:

            doc_id = result.get(
                "doc_id"
            )

            if doc_id is None:
                continue

            if doc_id not in candidates:

                candidates[doc_id] = {
                    "doc_id": doc_id,
                    "originals": [],
                    "routes": [],
                    "route_hits": {},
                    "entries": [],
                }

            candidate = candidates[doc_id]

            # ----------------------------------------------------------
            # Preserve original token evidence.
            # ----------------------------------------------------------

            original = result.get(
                "original"
            )

            if (
                original
                and original not in candidate["originals"]
            ):
                candidate["originals"].append(
                    original
                )

            # ----------------------------------------------------------
            # Preserve route evidence.
            # ----------------------------------------------------------

            if route not in candidate["routes"]:
                candidate["routes"].append(
                    route
                )

            candidate["route_hits"][route] = (
                candidate["route_hits"].get(
                    route,
                    0,
                ) + 1
            )

            # ----------------------------------------------------------
            # Preserve the actual retrieval entry.
            # ----------------------------------------------------------

            candidate["entries"].append(
                result
            )

    # ==================================================================
    # CONTEXT RETRIEVAL
    # ==================================================================

    def get_context(
        self,
        query: str,
        lang: str = "en",
        limit: int = 10,
    ) -> str:
        """
        Retrieve and rank context from MemoryGrid.

        The query enters through:

            Letter Grid
            Word Grid
            Full-Text Grid

        These remain independent during storage and retrieval.

        They are unified only when candidate documents are assembled
        for ranking.

        Ranking then determines which documents become context for the
        prompt/context architecture.
        """

        if not query:
            return ""

        lang = normalize_lang(
            lang
        )

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

        cached = self.memory_cache.get(
            query
        )

        if cached:
            return cached

        # --------------------------------------------------------------
        # Tokenize
        # --------------------------------------------------------------

        tokens = tokenize(
            query,
            lang,
        )

        if not tokens:
            self.memory_cache.set(
                query,
                "",
            )

            return ""

        # --------------------------------------------------------------
        # Directive
        # --------------------------------------------------------------

        directive = detect_directive(
            query
        )

        # --------------------------------------------------------------
        # Certain directive types can benefit from a slightly wider
        # retrieval window.
        # --------------------------------------------------------------

        crawl_limit = (
            limit + 2
            if directive in {
                "entity_identity",
                "location",
            }
            else limit
        )

        # --------------------------------------------------------------
        # THREE MEMORY ROUTES
        # --------------------------------------------------------------

        candidates = self._retrieve_three_routes(
            query=query,
            lang=lang,
            limit=crawl_limit,
        )

        if not candidates:

            self.memory_cache.set(
                query,
                "",
            )

            return ""

        # --------------------------------------------------------------
        # DOMAIN
        # --------------------------------------------------------------

        domain = detect_domain(
            query
        )

        # --------------------------------------------------------------
        # COLLECT DOCUMENTS
        # --------------------------------------------------------------

        candidate_docs = []

        for candidate in candidates:

            doc_id = candidate.get(
                "doc_id"
            )

            if doc_id is None:
                continue

            doc_text = self.memory.get_doc(
                doc_id
            )

            if not doc_text:
                continue

            candidate_docs.append({
                "doc_id": doc_id,
                "text": doc_text,
                "routes": candidate.get(
                    "routes",
                    [],
                ),
                "route_hits": candidate.get(
                    "route_hits",
                    {},
                ),
                "entries": candidate.get(
                    "entries",
                    [],
                ),
            })

        if not candidate_docs:

            self.memory_cache.set(
                query,
                "",
            )

            return ""

        # --------------------------------------------------------------
        # RANK CANDIDATE DOCUMENTS
        # --------------------------------------------------------------

        ranked = []

        for doc in candidate_docs:

            score_result = score_candidate(
                query=query,
                candidate_text=doc["text"],
                query_entities=[],
                query_hierarchy=domain,
                candidate_entities=[],
                candidate_hierarchy=domain,
                freshness_score=0.0,
                lang=lang,
            )

            ranked.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "score": score_result["total"],
                "scores": score_result["scores"],
                "routes": doc["routes"],
                "route_hits": doc["route_hits"],
            })

        # --------------------------------------------------------------
        # PRIMARY RANK
        #
        # The established ranking system remains authoritative.
        #
        # Route information is preserved as retrieval evidence rather
        # than being silently turned into an invented scoring formula.
        # --------------------------------------------------------------

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # --------------------------------------------------------------
        # TOP CONTEXT
        # --------------------------------------------------------------

        context = "\n".join(
            item["text"]
            for item in ranked[:3]
        )

        self.memory_cache.set(
            query,
            context,
        )

        return context

    # ==================================================================
    # LOCAL LEXICAL SCORE
    # ==================================================================

    def score_candidate(
        self,
        query: str,
        doc_text: str,
        lang: str = "en",
    ) -> float:
        """
        Calculate the local lexical similarity score.

        Letter and Word scoring remain separate.

            letter = 40%
            word   = 60%

        This method is separate from the global ranking.py scorer.
        """

        lang = normalize_lang(
            lang
        )

        q_tokens = tokenize(
            query,
            lang,
        )

        if not q_tokens:
            return 0.0

        l_score = letter_score(
            q_tokens,
            doc_text,
            lang,
        )

        w_score = word_score(
            q_tokens,
            doc_t