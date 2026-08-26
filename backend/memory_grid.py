from __future__ import annotations

from typing import List, Dict, Any, Optional, Iterable

from tokenizer import tokenize, normalize_lang


class MemoryGrid:
    """
    CoMpaNeoN knowledge memory.

    Knowledge enters the memory through three independent routes:

    1. Letter route
       - Uses tokenizer letter indices.
       - Language alphabet × 1.

    2. Word route
       - Uses tokenizer word-grid information.
       - Language alphabet × alphabet.

    3. Storage route
       - Uses the length of words contained in an entry.
       - Maps into the independent 64-row storage dimension.

    These routes are intentionally independent.

    User entries, crawled documents, research material, dictionary data,
    training data, and other indexed knowledge all use the same placement
    model.
    """

    def __init__(self, rows: int = 64, cols: int = 26):
        self.rows = rows
        self.cols = cols

        # ------------------------------------------------------------------
        # Three independent indexes
        # ------------------------------------------------------------------

        # Letter route:
        # {
        #     letter_index: [token records...]
        # }
        self.letter_grid: Dict[int, List[Dict[str, Any]]] = {}

        # Word route:
        # {
        #     (word_col, word_row): [token records...]
        # }
        self.word_grid: Dict[tuple, List[Dict[str, Any]]] = {}

        # Storage route:
        # {
        #     storage_row: [token records...]
        # }
        self.storage_grid: Dict[int, List[Dict[str, Any]]] = {}

        # Complete document store.
        self.doc_store: List[Dict[str, Any]] = []

    # ======================================================================
    # ROUTE 1 — LETTER GRID
    # ======================================================================

    def _letter_cells(
        self,
        token_info: Dict[str, Any],
    ) -> Iterable[int]:
        """
        Return the independent letter-grid positions for a token.

        The tokenizer has already resolved the language-specific alphabet
        index. We preserve those indices rather than deriving a row/column
        from them.
        """

        return token_info.get("letter", []) or []

    def _place_letter(
        self,
        letter_index: int,
        record: Dict[str, Any],
    ) -> None:
        """
        Place one token record into the letter route.
        """

        cell = int(letter_index)

        if cell not in self.letter_grid:
            self.letter_grid[cell] = []

        self.letter_grid[cell].append(record)

    def get_tokens_at_letter(
        self,
        letter_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve records from a letter-grid position.
        """

        return list(
            self.letter_grid.get(
                int(letter_index),
                [],
            )
        )

    # ======================================================================
    # ROUTE 2 — WORD GRID
    # ======================================================================

    def _word_cell(
        self,
        token_info: Dict[str, Any],
    ) -> tuple:
        """
        Return the tokenizer's A×A word-grid coordinate.

        This is NOT the 64-row storage route.
        """

        word = token_info.get("word") or {}

        col = int(
            word.get("col", 0)
        )

        row = int(
            word.get("row", 0)
        )

        return row, col

    def _place_word(
        self,
        token_info: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """
        Place a token into the language word-grid route.
        """

        row, col = self._word_cell(
            token_info
        )

        cell = (row, col)

        if cell not in self.word_grid:
            self.word_grid[cell] = []

        self.word_grid[cell].append(record)

    def get_tokens_at_word(
        self,
        row: int,
        col: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve records from the word-grid coordinate.

        The tokenizer's A×A coordinate is preserved.
        """

        return list(
            self.word_grid.get(
                (int(row), int(col)),
                [],
            )
        )

    # ======================================================================
    # ROUTE 3 — 64-ROW STORAGE GRID
    # ======================================================================

    def _storage_cells(
        self,
        token_info: Dict[str, Any],
    ) -> List[int]:
        """
        Determine the independent 64-row storage position.

        Storage is based on the length of the word contained in the entry.

        This route is deliberately separate from:
            - letter-grid indexing
            - word-grid indexing
        """

        word = token_info.get("original", "")

        if not word:
            return []

        length = len(word)

        row = length % self.rows

        return [row]

    def _place_storage(
        self,
        token_info: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """
        Place a token into the independent 64-row storage route.
        """

        for row in self._storage_cells(
            token_info
        ):
            if row not in self.storage_grid:
                self.storage_grid[row] = []

            self.storage_grid[row].append(record)

    def get_tokens_at_storage(
        self,
        row: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve records from the 64-row storage route.
        """

        row = int(row) % self.rows

        return list(
            self.storage_grid.get(
                row,
                [],
            )
        )

    # ======================================================================
    # DOCUMENT INSERTION
    # ======================================================================

    def add_document(
        self,
        text: str,
        lang: str = "en",
        source: str = "",
    ) -> int:
        """
        Add an entry to CoMpaNeoN memory.

        Every token is entered through all applicable independent routes.
        """

        lang = normalize_lang(lang)

        tokens = tokenize(
            text,
            lang,
        )

        doc_id = len(
            self.doc_store
        )

        self.doc_store.append({
            "text": text,
            "source": source,
            "lang": lang,
            "tokens": tokens,
        })

        for token in tokens:

            record = {
                "doc_id": doc_id,
                "original": token.get("original", ""),
                "stem": token.get("stem", ""),
                "word": token.get("word", {}),
                "letter": token.get("letter", []),
                "lang": token.get("lang", lang),
                "source": source,
            }

            # --------------------------------------------------------------
            # Route 1 — Letter
            # --------------------------------------------------------------

            for letter_index in self._letter_cells(
                token
            ):
                self._place_letter(
                    letter_index,
                    record,
                )

            # --------------------------------------------------------------
            # Route 2 — Word
            # --------------------------------------------------------------

            self._place_word(
                token,
                record,
            )

            # --------------------------------------------------------------
            # Route 3 — 64-row storage
            # --------------------------------------------------------------

            self._place_storage(
                token,
                record,
            )

        return doc_id

    # ======================================================================
    # MULTI-ROUTE RETRIEVAL
    # ======================================================================

    def retrieve_by_token(
        self,
        token_info: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve candidates through all three routes for one token.

        No ranking is performed here.
        """

        result = {
            "letter": [],
            "word": [],
            "storage": [],
        }

        # Letter route
        seen_letter = set()

        for letter_index in self._letter_cells(
            token_info
        ):
            for item in self.get_tokens_at_letter(
                letter_index
            ):
                key = (
                    item.get("doc_id"),
                    item.get("original"),
                )

                if key in seen_letter:
                    continue

                seen_letter.add(key)
                result["letter"].append(item)

        # Word route
        word_row, word_col = self._word_cell(
            token_info
        )

        seen_word = set()

        for item in self.get_tokens_at_word(
            word_row,
            word_col,
        ):
            key = (
                item.get("doc_id"),
                item.get("original"),
            )

            if key in seen_word:
                continue

            seen_word.add(key)
            result["word"].append(item)

        # Storage route
        seen_storage = set()

        for row in self._storage_cells(
            token_info
        ):
            for item in self.get_tokens_at_storage(
                row
            ):
                key = (
                    item.get("doc_id"),
                    item.get("original"),
                )

                if key in seen_storage:
                    continue

                seen_storage.add(key)
                result["storage"].append(item)

        return result

    def retrieve(
        self,
        token_infos: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve candidates for multiple query tokens.

        Results remain separated by route.
        """

        result = {
            "letter": [],
            "word": [],
            "storage": [],
        }

        seen = {
            "letter": set(),
            "word": set(),
            "storage": set(),
        }

        for token_info in token_infos:

            routes = self.retrieve_by_token(
                token_info
            )

            for route in result:

                for item in routes[route]:

                    key = (
                        item.get("doc_id"),
                        item.get("original"),
                    )

                    if key in seen[route]:
                        continue

                    seen[route].add(key)
                    result[route].append(item)

        return result

    # ======================================================================
    # DOCUMENT ACCESS
    # ======================================================================

    def get_doc(
        self,
        doc_id: int,
    ) -> str:
        if 0 <= doc_id < len(
            self.doc_store
        ):
            return self.doc_store[doc_id]["text"]

        return ""

    def get_doc_tokens(
        self,
        doc_id: int,
    ) -> list:
        if 0 <= doc_id < len(
            self.doc_store
        ):
            return self.doc_store[doc_id]["tokens"]

        return []

    def get_doc_source(
        self,
        doc_id: int,
    ) -> str:
        if 0 <= doc_id < len(
            self.doc_store
        ):
            return self.doc_store[doc_id].get(
                "source",
                "",
            )

        return ""

    def get_doc_language(
        self,
        doc_id: int,
    ) -> str:
        if 0 <= doc_id < len(
            self.doc_store
        ):
            return self.doc_store[doc_id].get(
                "lang",
                "en",
            )

        return ""

    # ======================================================================
    # LEGACY-COMPATIBLE ACCESS
    # ======================================================================

    def get_tokens_at(
        self,
        row: int,
        col: int,
    ) -> List[Dict[str, Any]]:
        """
        Compatibility accessor for callers that previously expected
        get_tokens_at(row, col).

        This now means WORD GRID access.

        New code should explicitly use:
            get_tokens_at_letter()
            get_tokens_at_word()
            get_tokens_at_storage()
        """

        return self.get_tokens_at_word(
            row,
            col,
        )