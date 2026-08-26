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
    CoMpaNeoN Word Understanding Layer.

    This layer understands a query by entering the MemoryGrid through
    three independent routes:

        1. Letter Grid
        2. Word Grid
        3. Full-text / Storage Grid

    The routes remain independent during retrieval.

    They are unified only after retrieval, when candidate documents
    are collected and ranked.

    ---------------------------------------------------------------
    LETTER GRID
    ---------------------------------------------------------------

        Language alphabet × 1

        Uses tokenizer letter mappings.

    ---------------------------------------------------------------
    WORD GRID
    ---------------------------------------------------------------

        Language alphabet × alphabet

        Complete word mapping.

        Word-grid rule:

            L   = name length
            uID = L
            S   = L

        Therefore:

            S = L

    ---------------------------------------------------------------
    FULL-TEXT / STORAGE GRID
    ---------------------------------------------------------------

        Independent 64-row storage mapping.

        This is handled by MemoryGrid.

        WordUnderstanding does NOT recreate its placement formula.

    ---------------------------------------------------------------

    The module also attaches:

        - directives
        - domain
        - symbols
        - code terminology
        - lexical scores
        - retrieved context

    It does not replace the ranking architecture.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

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
        Produce a deterministic hash for text.

        Kept available for cache/context workflows.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ==================================================================
    # LETTER ROUTE
    # ==================================================================

    def _retrieve_letter_route(
        self,
        token: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates through the Letter Grid.

        The tokenizer has already resolved the token's letters into
        alphabet indices.

        This route does not calculate:

            L + S
            storage rows
            word-grid coordinates

        It simply follows the letter mappings.
        """

        results: List[
            Dict[str, Any]
        ] = []

        seen = set()

        letters = token.get(
            "letter",
            [],
        ) or []

        for letter_index in letters:

            entries = (
                self.memory.get_tokens_at_letter(
                    letter_index
                )
            )

            for entry in entries:

                key = (
                    entry.get("doc_id"),
                    entry.get("original"),
                )

                if key in seen:
                    continue

                seen.add(key)

                result = dict(
                    entry
                )

                result[
                    "retrieval_route"
                ] = "letter"

                results.append(
                    result
                )

        return results

    # ==================================================================
    # WORD ROUTE
    # ==================================================================

    def _retrieve_word_route(
        self,
        token: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the complete word through the A×A Word Grid.

        Word Grid rules are established by tokenizer.py:

            L   = name length
            uID = L
            S   = L

        Therefore S is equal to L.

        The tokenizer resolves the word into its A×A coordinate.

        WordUnderstanding does not reinterpret that coordinate as
        a 64-row storage coordinate.
        """

        word = token.get(
            "word"
        )

        if not word:
            return []

        row = int(
            word.get(
                "row",
                0,
            )
        )

        col = int(
            word.get(
                "col",
                0,
            )
        )

        entries = (
            self.memory.get_tokens_at_word(
                row,
                col,
            )
        )

        results: List[
            Dict[str, Any]
        ] = []

        for entry in entries:

            result = dict(
                entry
            )

            result[
                "retrieval_route"
            ] = "word"

            results.append(
                result
            )

        return results

    # ==================================================================
    # FULL-TEXT / STORAGE ROUTE
    # ==================================================================

    def _retrieve_storage_route(
        self,
        token: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve through the independent 64-row storage route.

        The storage mapping belongs to MemoryGrid.

        WordUnderstanding therefore does not reproduce the storage
        formula here.

        This is important because:

            Word Grid != Storage Grid

        The storage route is based on the independent storage
        dimension established by MemoryGrid.
        """

        routes = (
            self.memory.retrieve_by_token(
                token
            )
        )

        entries = routes.get(
            "storage",
            [],
        )

        results: List[
            Dict[str, Any]
        ] = []

        for entry in entries[:limit]:

            result = dict(
                entry
            )

            result[
                "retrieval_route"
            ] = "storage"

            results.append(
                result
            )

        return results

    # ==================================================================
    # THREE-ROUTE RETRIEVAL
    # ==================================================================

    def _retrieve_three_routes(
        self,
        query: str,
        lang: str,
        limit: int,
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:
        """
        Enter the query into all three MemoryGrid routes.

        The routes remain separate in the returned structure.

        Returns:

            {
                "letter": [...],
                "word": [...],
                "storage": [...]
            }

        No ranking occurs here.
        """

        tokens = tokenize(
            query,
            lang,
        )

        routes = {
            "letter": [],
            "word": [],
            "storage": [],
        }

        seen = {
            "letter": set(),
            "word": set(),
            "storage": set(),
        }

        for token in tokens:

            # ----------------------------------------------------------
            # ROUTE 1 — LETTER
            # ----------------------------------------------------------

            letter_results = (
                self._retrieve_letter_route(
                    token
                )
            )

            for result in letter_results:

                key = (
                    result.get("doc_id"),
                    result.get("original"),
                )

                if key in seen["letter"]:
                    continue

                seen["letter"].add(
                    key
                )

                routes[
                    "letter"
                ].append(
                    result
                )

            # ----------------------------------------------------------
            # ROUTE 2 — WORD
            # ----------------------------------------------------------

            word_results = (
                self._retrieve_word_route(
                    token
                )
            )

            for result in word_results:

                key = (
                    result.get("doc_id"),
                    result.get("original"),
                )

                if key in seen["word"]:
                    continue

                seen["word"].add(
                    key
                )

                routes[
                    "word"
                ].append(
                    result
                )

            # ----------------------------------------------------------
            # ROUTE 3 — STORAGE
            # ----------------------------------------------------------

            storage_results = (
                self._retrieve_storage_route(
                    token,
                    limit,
                )
            )

            for result in storage_results:

                key = (
                    result.get("doc_id"),
                    result.get("original"),
                )

                if key in seen["storage"]:
                    continue

                seen["storage"].add(
                    key
                )

                routes[
                    "storage"
                ].append(
                    result
                )

        return routes

    # ==================================================================
    # CANDIDATE DOCUMENT COLLECTION
    # ==================================================================

    def _collect_candidate_documents(
        self,
        routes: Dict[
            str,
            List[Dict[str, Any]]
        ],
    ) -> Dict[
        int,
        Dict[str, Any]
    ]:
        """
        Convert route-level token candidates into unique documents.

        Route identity is preserved in the candidate metadata.

        A document can be discovered by multiple routes.
        """

        candidate_docs: Dict[
            int,
            Dict[str, Any]
        ] = {}

        for route_name in (
            "letter",
            "word",
            "storage",
        ):

            for result in routes.get(
                route_name,
                [],
            ):

                doc_id = result.get(
                    "doc_id"
                )

                if doc_id is None:
                    continue

                if doc_id not in candidate_docs:

                    doc_text = self.memory.get_doc(
                        doc_id
                    )

                    if not doc_text:
                        continue

                    candidate_docs[
                        doc_id
                    ] = {
                        "doc_id": doc_id,
                        "text": doc_text,
                        "routes": set(),
                    }

                candidate_docs[
                    doc_id
                ][
                    "routes"
                ].add(
                    route_name
                )

        return candidate_docs

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
        Retrieve and rank contextual knowledge for a query.

        Flow:

            Query
              │
              ├── Letter Grid
              │
              ├── Word Grid
              │
              └── Storage Grid
                       │
                       ▼
                Candidate Documents
                       │
                       ▼
                    Ranking
                       │
                       ▼
                 Top Context

        The three physical/logical memory routes remain separate.
        Only their candidates are unified for ranking.
        """

        cached = self.memory_cache.get(
            query
        )

        if cached:
            return cached

        lang = normalize_lang(
            lang
        )

        tokens = tokenize(
            query,
            lang,
        )

        if not tokens:
            return ""

        # --------------------------------------------------------------
        # DIRECTIVE
        # --------------------------------------------------------------

        directive = detect_directive(
            query
        )

        # --------------------------------------------------------------
        # RETRIEVAL LIMIT
        # --------------------------------------------------------------

        crawl_limit = limit

        if directive in {
            "entity_identity",
            "location",
        }:
            crawl_limit = (
                limit + 2
            )

        # --------------------------------------------------------------
        # THREE ROUTES
        # --------------------------------------------------------------

        routes = (
            self._retrieve_three_routes(
                query=query,
                lang=lang,
                limit=crawl_limit,
            )
        )

        # --------------------------------------------------------------
        # COLLECT DOCUMENTS
        # --------------------------------------------------------------

        candidate_docs = (
            self._collect_candidate_documents(
                routes
            )
        )

        if not candidate_docs:

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
        # RANK
        # --------------------------------------------------------------

        ranked: List[
            Dict[str, Any]
        ] = []

        for doc in candidate_docs.values():

            route_names = sorted(
                doc.get(
                    "routes",
                    set(),
                )
            )

            # ----------------------------------------------------------
            # Candidate scoring
            # ----------------------------------------------------------

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
                "routes": route_names,
            })

        # --------------------------------------------------------------
        # SORT
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
    # LEXICAL CANDIDATE SCORE
    # ==================================================================

    def score_candidate(
        self,
        query: str,
        doc_text: str,
        lang: str = "en",
    ) -> float:
        """
        Calculate the local lexical score.

        The tokenizer provides:

            letter mappings
            word mappings

        Letter and word similarity remain independent components.

        This method does not replace the global ranking.score_candidate()
        system.
        """

        lang = normalize_lang(
            lang
        )

        query_tokens = tokenize(
            query,
            lang,
        )

        if not query_tokens:
            return 0.0

        letter_component = letter_score(
            query_tokens,
            doc_text,
            lang,
        )

        word_component = word_score(
            query_tokens,
            doc_text,
            lang,
        )

        return (
            letter_component * 0.4
            + word_component * 0.6
        )

    # ==================================================================
    # EXPLICIT DOCUMENT RANKING
    # ==================================================================

    def rank_documents(
        self,
        query: str,
        doc_ids: List[int],
        lang: str = "en",
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Rank explicitly supplied documents using the local lexical
        scoring layer.
        """

        lang = normalize_lang(
            lang
        )

        scored: List[
            Dict[str, Any]
        ] = []

        for doc_id in doc_ids:

            text = self.memory.get_doc(
                doc_id
            )

            if not text:
                continue

            score = self.score_candidate(
                query=query,
                doc_text=text,
                lang=lang,
            )

            scored.append({
                "doc_id": doc_id,
                "score": score,
                "text": text,
            })

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored

    # ==================================================================
    # QUERY ENRICHMENT
    # ==================================================================

    def _build_enrichment(
        self,
        query: str,
        domain: str,
    ) -> Dict[str, Any]:
        """
        Extract query-level enrichment without changing the query.

        Includes:

            symbols
            code terminology
            directive
            domain
        """

        enrichment: Dict[
            str,
            Any
        ] = {
            "symbols": [],
            "code_terms": [],
            "directive": detect_directive(
                query
            ),
            "domain": domain,
        }

        # --------------------------------------------------------------
        # SYMBOLS
        # --------------------------------------------------------------

        enr