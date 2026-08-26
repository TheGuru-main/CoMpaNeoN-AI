"""
Word Understanding Module

CoMpaNeoN AI understanding layer.

Three independent memory/grid entry routes:

1. WORD TOKEN ROUTE
   Query word tokens enter through the 26-column word-token grid.
   Word token -> mod 26.

2. LETTER TOKEN ROUTE
   Query letter tokens enter through the letter-token route.
   Letter token -> mod 1.
   This route is independent of the word-token route.

3. USER-ENTRY LONG-WORD STORAGE ROUTE
   Long words found in the user's stored entries are treated as
   memory-storage words, not query tokens.
   Their storage route uses mod 64.
   This route is independent from tokenization.

The three routes converge only at candidate collection/ranking.

Main GSP placement remains separate.
Project/workspace context remains separate.
Directive detection remains separate.
Domain detection remains separate.

The purpose of this module is to collect the strongest available
context for Prompt Manager / CoMpaNeoN AI without replacing any
existing architecture.
"""

import hashlib
from typing import List, Dict, Any, Optional, Iterable

from tokenizer import (
    tokenize,
    normalize_lang,
    letter_score,
    word_score,
    supported_languages,
)

from memory_grid import MemoryGrid
from grid_crawler import crawl as grid_crawl

from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive

from page_cache import PageCache
from memory_cache import MemoryCache

from ranking import score_candidate
from intent_analyzer import detect_domain


# ---------------------------------------------------------------------------
# GRID CONSTANTS
# ---------------------------------------------------------------------------

WORD_GRID_COLUMNS = 26
LETTER_GRID_MODULUS = 1
STORAGE_ROWS = 64

# Words above this length are treated as long-word storage candidates.
# This does NOT alter the tokenizer.
LONG_WORD_LENGTH = 8


class WordUnderstanding:

    def __init__(self, memory_grid: MemoryGrid):
        self.memory = memory_grid

        self.page_cache = PageCache()
        self.memory_cache = MemoryCache()

    # =======================================================================
    # BASIC HELPERS
    # =======================================================================

    def _hash(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_word(self, word: str) -> str:
        return (
            str(word)
            .strip()
            .lower()
        )

    # =======================================================================
    # ROUTE 1
    # WORD TOKEN -> MOD 26
    # =======================================================================

    def word_token_column(
        self,
        token: Dict[str, Any],
    ) -> int:
        """
        Determine the word-token column.

        This route is strictly word-token based.

        The tokenizer's word token provides the column value.
        The result is normalized through mod 26.

        No row is generated here.
        No mod 64 is performed here.
        """

        word_data = token.get("word", token)

        raw_column = word_data.get(
            "col",
            word_data.get("column", 0)
        )

        column = self._safe_int(
            raw_column
        )

        return column % WORD_GRID_COLUMNS

    # =======================================================================
    # ROUTE 2
    # LETTER TOKEN -> MOD 1
    # =======================================================================

    def letter_token_route(
        self,
        token: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine the independent letter-token route.

        Letter-token routing is deliberately separate from word-token
        routing.

        The letter route uses mod 1.

        No mod 26 is applied here.
        No mod 64 is applied here.
        """

        letters = []

        # Different tokenizer versions may expose letters differently.
        raw_letters = token.get("letters")

        if raw_letters:
            if isinstance(raw_letters, str):
                letters = list(raw_letters)
            elif isinstance(raw_letters, Iterable):
                letters = list(raw_letters)

        # If the tokenizer exposes an original token but no explicit
        # letter list, preserve the letters directly for this route.
        if not letters:
            original = (
                token.get("original")
                or token.get("text")
                or token.get("token")
                or ""
            )

            if isinstance(original, str):
                letters = list(original)

        routes = []

        for letter in letters:
            if not letter:
                continue

            # Letter token itself is preserved.
            # Its routing modulus is explicitly 1.
            routes.append({
                "letter": letter,
                "mod": LETTER_GRID_MODULUS,
                "route": "letter_token",
            })

        return {
            "route": "letter_token",
            "modulus": LETTER_GRID_MODULUS,
            "letters": routes,
        }

    # =======================================================================
    # ROUTE 3
    # USER ENTRY LONG WORD -> MOD 64
    # =======================================================================

    def long_word_storage_row(
        self,
        word: str,
    ) -> Optional[int]:
        """
        Map a long word from user-entry storage to the 64-row storage
        space.

        IMPORTANT:

        This is NOT the query word-token route.

        It exists specifically for long words originating from stored
        user entries.

        The word length is used for the 64-row storage mapping.
        """

        normalized = self._normalize_word(
            word
        )

        if len(normalized) < LONG_WORD_LENGTH:
            return None

        return len(normalized) % STORAGE_ROWS

    def extract_long_words_from_user_entries(
        self,
        entries: Iterable[Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract long words from stored user entries.

        Entries may be:

        - Message objects
        - dictionaries containing 'content'
        - strings

        This function does not use query token positions.
        """

        results = []
        seen = set()

        for entry in entries or []:

            if isinstance(entry, str):
                text = entry

            elif isinstance(entry, dict):
                text = (
                    entry.get("content")
                    or entry.get("text")
                    or ""
                )

            else:
                text = getattr(
                    entry,
                    "content",
                    ""
                )

            if not text:
                continue

            try:
                tokens = tokenize(
                    text,
                    "en",
                )
            except Exception:
                tokens = []

            # Prefer tokenizer words when available.
            words = []

            for token in tokens:
                word_data = token.get(
                    "word",
                    {}
                )

                word = (
                    word_data.get("original")
                    or word_data.get("word")
                    or token.get("original")
                    or token.get("text")
                    or token.get("token")
                )

                if word:
                    words.append(
                        str(word)
                    )

            # Safe fallback when tokenizer does not expose word fields.
            if not words:
                words = text.split()

            for word in words:

                normalized = self._normalize_word(
                    word
                )

                if len(normalized) < LONG_WORD_LENGTH:
                    continue

                if normalized in seen:
                    continue

                seen.add(normalized)

                row = self.long_word_storage_row(
                    normalized
                )

                if row is None:
                    continue

                results.append({
                    "word": normalized,
                    "length": len(normalized),
                    "row": row,
                    "mod": STORAGE_ROWS,
                    "route": "user_entry_long_word",
                })

        return results

    # =======================================================================
    # TOKENIZATION
    # =======================================================================

    def _tokenize_query(
        self,
        query: str,
        lang: str,
    ) -> List[Dict[str, Any]]:

        try:
            return tokenize(
                query,
                lang,
            )
        except Exception:
            return []

    # =======================================================================
    # ROUTE 1 CRAWL
    # =======================================================================

    def _crawl_word_token_route(
        self,
        tokens: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Crawl memory through the word-token / 26-column route.

        This route does not calculate a 64-row location.

        The crawler receives the column route independently.
        """

        results = []

        seen = set()

        for token in tokens:

            column = self.word_token_column(
                token
            )

            # The existing crawler requires a row.
            # Do not manufacture a GSP row here.
            #
            # The route is therefore represented explicitly and
            # handed to the crawler only when the underlying memory
            # implementation supports column-first traversal.

            try:
                token_results = grid_crawl(
                    self.memory,
                    None,
                    column,
                    limit=limit,
                    route="word_token",
                )
            except TypeError:
                # Backward compatibility with the existing crawler
                # signature. The route information remains attached
                # to the result collection.
                token_results = []

            for result in token_results:

                result = dict(result)

                result["_understanding_route"] = (
                    "word_token"
                )

                result["_word_column"] = column

                key = (
                    result.get("doc_id"),
                    result.get("original"),
                    "word_token",
                    column,
                )

                if key in seen:
                    continue

                seen.add(key)
                results.append(result)

        return results

    # =======================================================================
    # ROUTE 2 CRAWL
    # =======================================================================

    def _crawl_letter_token_route(
        self,
        tokens: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Crawl memory through the independent letter-token route.

        The route is intentionally not merged with the word-token
        column calculation.
        """

        results = []

        seen = set()

        for token in tokens:

            letter_route = self.letter_token_route(
                token
            )

            for letter_data in letter_route["letters"]:

                letter = letter_data["letter"]

                try:
                    token_results = grid_crawl(
                        self.memory,
                        None,
                        None,
                        limit=limit,
                        route="letter_token",
                        letter=letter,
                        modulus=LETTER_GRID_MODULUS,
                    )
                except TypeError:
                    token_results = []

                for result in token_results:

                    result = dict(result)

                    result["_understanding_route"] = (
                        "letter_token"
                    )

                    result["_letter"] = letter

                    key = (
                        result.get("doc_id"),
                        result.get("original"),
                        "letter_token",
                        letter,
                    )

                    if key in seen:
                        continue

                    seen.add(key)
                    results.append(result)

        return results

    # =======================================================================
    # ROUTE 3 CRAWL
    # =======================================================================

    def _crawl_user_entry_long_word_route(
        self,
        user_entries: Iterable[Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Crawl memory through long words originating from user entries.

        This is the only understanding route that uses mod 64.

        It is not derived from the query word token.
        """

        long_words = (
            self.extract_long_words_from_user_entries(
                user_entries
            )
        )

        results = []

        seen = set()

        for item in long_words:

            row = item["row"]

            try:
                token_results = grid_crawl(
                    self.memory,
                    row,
                    None,
                    limit=limit,
                    route="user_entry_long_word",
                    word=item["word"],
                    modulus=STORAGE_ROWS,
                )
            except TypeError:
                token_results = []

            for result in token_results:

                result = dict(result)

                result["_understanding_route"] = (
                    "user_entry_long_word"
                )

                result["_storage_word"] = (
                    item["word"]
                )

                result["_storage_row"] = row

                key = (
                    result.get("doc_id"),
                    result.get("original"),
                    "user_entry_long_word",
                    item["word"],
                    row,
                )

                if key in seen:
                    continue

                seen.add(key)
                results.append(result)

        return results

    # =======================================================================
    # CANDIDATE DOCUMENT COLLECTION
    # =======================================================================

    def _collect_candidate_documents(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        candidate_docs = []

        seen_doc_ids = set()

        for result in results:

            doc_id = result.get(
                "doc_id"
            )

            if doc_id is None:
                continue

            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(
                doc_id
            )

            doc_text = self.memory.get_doc(
                doc_id
            )

            if not doc_text:
                continue

            candidate_docs.append({
                "doc_id": doc_id,
                "text": doc_text,
                "routes": [
                    result.get(
                        "_understanding_route"
                    )
                ],
            })

        return candidate_docs

    # =======================================================================
    # CONTEXT RETRIEVAL
    # =======================================================================

    def get_context(
        self,
        query: str,
        lang: str = "en",
        limit: int = 10,
        user_entries: Optional[Iterable[Any]] = None,
    ) -> str:
        """
        Retrieve and rank context from the three independent routes.

        Routes:

            word token -> mod 26
            letter token -> mod 1
            user-entry long word -> mod 64

        The routes converge only after retrieval.
        """

        cache_key = (
            f"{query}|{normalize_lang(lang)}"
        )

        cached = self.memory_cache.get(
            cache_key
        )

        if cached:
            return cached

        tokens = self._tokenize_query(
            query,
            lang,
        )

        if not tokens:
            return ""

        directive = detect_directive(
            query
        )

        crawl_limit = (
            limit + 2
            if directive in {
                "entity_identity",
                "location",
            }
            else limit
        )

        # ---------------------------------------------------------------
        # THREE INDEPENDENT ENTRY ROUTES
        # ---------------------------------------------------------------

        results = []

        # Route 1: word token -> mod 26
        word_results = (
            self._crawl_word_token_route(
                tokens,
                crawl_limit,
            )
        )

        results.extend(
            word_results
        )

        # Route 2: letter token -> mod 1
        letter_results = (
            self._crawl_letter_token_route(
                tokens,
                crawl_limit,
            )
        )

        results.extend(
            letter_results
        )

        # Route 3: user's stored long words -> mod 64
        if user_entries:
            storage_results = (
                self._crawl_user_entry_long_word_route(
                    user_entries,
                    crawl_limit,
                )
            )

            results.extend(
                storage_results
            )

        # ---------------------------------------------------------------
        # NO CANDIDATES
        # ---------------------------------------------------------------

        candidate_docs = (
            self._collect_candidate_documents(
                results
            )
        )

        if not candidate_docs:

            self.memory_cache.set(
                cache_key,
                "",
            )

            return ""

        # ---------------------------------------------------------------
        # DOMAIN
        # ---------------------------------------------------------------

        domain = detect_domain(
            query
        )

        # ---------------------------------------------------------------
        # RANK
        # ---------------------------------------------------------------

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
            })

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # ---------------------------------------------------------------
        # TOP CONTEXT
        # ---------------------------------------------------------------

        context = "\n".join(
            item["text"]
            for item in ranked[:3]
        )

        self.memory_cache.set(
            cache_key,
            context,
        )

        return context

    # =======================================================================
    # LEXICAL CANDIDATE SCORE
    # =======================================================================

    def score_candidate(
        self,
        query: str,
        doc_text: str,
        lang: str = "en",
    ) -> float:
        """
        Return the local lexical score for a candidate.

        This remains separate from the higher-level candidate ranking
        performed by ranking.score_candidate().
        """

        q_tokens = self._tokenize_query(
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
            doc_text,
            lang,
        )

        return (
            l_score * 0.4
            + w_score * 0.6
        )

    # =======================================================================
    # EXPLICIT DOCUMENT RANKING
    # =======================================================================

    def rank_documents(
        self,
        query: str,
        doc_ids: List[int],
        lang: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Rank explicitly supplied documents.
        """

        scored = []

        for doc_id in doc_ids:

            text = self.memory.get_doc(
                doc_id
            )

            if not text:
                continue

            score = self.score_candidate(
                query,
                text,
                lang,
            )

            scored.append({
                "doc_id": doc_id,
                "score": score,
                "text": text,
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return scored

    # =======================================================================
    # FULL UNDERSTANDING OBJECT
    # =======================================================================

    def understand(
        self,
        query: str,
        lang: str = "en",
        user_entries: Optional[Iterable[Any]] = None,
    ) -> dict:
        """
        Produce the structured understanding state.

        This is the bridge between retrieval and Prompt Manager.

        It exposes the three routes explicitly so the upper architecture
        can use them without confusing their responsibilities.
        """

        tokens = self._tokenize_query(
            query,
            lang,
        )

        domain = detect_domain(
            query
        )

        symbols = recognize_symbols(
            query,
            domain,
        )

        directive = detect_directive(
            query
        )

        word_routes = []

        for token in tokens:

            word_routes.append({
                "word": (
                    token.get("word", {})
                    .get(
                        "original",
                        token.get(
                            "original",
                            token.get(
                                "text",
                                ""
                            )
                        )
                    )
                ),
                "column": self.word_token_column(
                    token
                ),
                "mod": WORD_GRID_COLUMNS,
            })

        letter_routes = []

        for token in tokens:

            letter_routes.append(
                self.letter_token_route(
                    token
                )
            )

        storage_routes = []

        if user_entries:

            storage_routes = (
                self.extract_long_words_from_user_entries(
                    user_entries
                )
            )

        context = self.get_context(
            query=query,
            lang=lang,
            user_entries=user_entries,
        )

        return {
            "context": context,

            "tokens": tokens,

            "domain": domain,

            "directive": directive,

            "symbols": symbols,

            "language": normalize_lang(
                lang
            ),

            "grid_routes": {
                "word_token": {
                    "modulus": WORD_GRID_COLUMNS,
                    "routes": word_routes,
                },

                "letter_token": {
                    "modulus": LETTER_GRID_MODULUS,
                    "routes": letter_routes,
                },

                "user_entry_long_word": {
                    "modulus": STORAGE_ROWS,
                    "routes": storage_routes,
                },
            },
        }

    # =======================================================================
    # PROJECT / WORKSPACE CONTEXT SUPPORT
    # =======================================================================

    def understand_project(
        self,
        query: str,
        project_name: str,
        messages: Optional[Iterable[Any]] = None,
        lang: str = "en",
    ) -> dict:
        """
        Project-aware understanding.

        Workspace/project messages are supplied as user-entry memory.
        They therefore participate in the third route only when their
        words qualify as long-word storage entries.

        The query itself continues to use the independent word-token
        and letter-token routes.
        """

        messages = list(
            messages or []
        )

        result = self.understand(
            query=query,
            lang=lang,
            user_entries=messages,
        )

        result["project"] = {
            "name": project_name,
            "message_count": len(messages),
        }

        return result