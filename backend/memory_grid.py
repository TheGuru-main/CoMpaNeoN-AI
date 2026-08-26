from __future__ import annotations

from typing import (
    List,
    Dict,
    Any,
    Iterable,
)

from tokenizer import (
    tokenize,
    normalize_lang,
)


class MemoryGrid:
    """
    CoMpaNeoN Knowledge Memory Grid.

    Knowledge may come from:
        - user entries
        - crawled documents
        - background training
        - dictionary/domain knowledge
        - research material
        - other indexed sources

    Every entry uses the same three independent memory routes:

    ---------------------------------------------------------------
    ROUTE 1 — LETTER GRID
    ---------------------------------------------------------------

        Language alphabet × 1

        Each letter token resolves directly to its alphabet index.

        Example:

            a -> 0
            b -> 1
            c -> 2
            ...

        The letter route does NOT calculate a row or word coordinate.

    ---------------------------------------------------------------
    ROUTE 2 — WORD GRID
    ---------------------------------------------------------------

        Language alphabet × alphabet

        The complete word itself is resolved.

        Word-grid rules:

            L   = name/word length
            uID = L
            S   = L

        Therefore:

            S = L

        The tokenizer resolves the word into its A×A word cell.

        This route is independent of the 64-row storage route.

    ---------------------------------------------------------------
    ROUTE 3 — FULL-TEXT / STORAGE GRID
    ---------------------------------------------------------------

        Independent 64-row storage dimension.

        This route is based on the length of words contained in
        the indexed entry.

        It does NOT use the Word Grid's A×A coordinate.

        It does NOT treat the word-grid row as the storage row.

        The 64-row storage dimension therefore remains independent
        from the letter and word routes.

    ---------------------------------------------------------------

    The three routes are unified only during retrieval/ranking.

    Storage placement and retrieval remain separate from lexical
    word-grid placement.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        rows: int = 64,
        cols: int = 26,
    ):
        """
        Initialize the memory grid.

        rows:
            Number of independent storage rows.

        cols:
            Default alphabet/column dimension.

        The default dimensions remain compatible with the established
        26-letter English grid while tokenizer.py remains responsible
        for language-specific alphabet dimensions.
        """

        self.rows = int(rows)
        self.cols = int(cols)

        # --------------------------------------------------------------
        # ROUTE 1
        # Letter Grid
        #
        # letter_index -> token records
        # --------------------------------------------------------------

        self.letter_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------
        # ROUTE 2
        # Word Grid
        #
        # (word_row, word_col) -> token records
        #
        # The tokenizer supplies the language-specific A×A coordinate.
        # --------------------------------------------------------------

        self.word_grid: Dict[
            tuple,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------
        # ROUTE 3
        # Full-text / Storage Grid
        #
        # storage_row -> token records
        #
        # This is the independent 64-row dimension.
        # --------------------------------------------------------------

        self.storage_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------
        # Complete document store
        # --------------------------------------------------------------

        self.doc_store: List[
            Dict[str, Any]
        ] = []

    # ==================================================================
    # ROUTE 1 — LETTER GRID
    # ==================================================================

    def _letter_cells(
        self,
        token_info: Dict[str, Any],
    ) -> Iterable[int]:
        """
        Return the letter-grid positions belonging to a token.

        tokenizer.py has already resolved each character into its
        language-specific alphabet index.

        We preserve those mappings exactly.

        No word-length calculation occurs here.
        No 64-row calculation occurs here.
        """

        return token_info.get(
            "letter",
            [],
        ) or []

    def _place_letter(
        self,
        letter_index: int,
        record: Dict[str, Any],
    ) -> None:
        """
        Place a token record into one letter-grid cell.
        """

        cell = int(letter_index)

        if cell not in self.letter_grid:
            self.letter_grid[cell] = []

        self.letter_grid[cell].append(
            record
        )

    def get_tokens_at_letter(
        self,
        letter_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve token records from a letter-grid position.
        """

        cell = int(letter_index)

        return list(
            self.letter_grid.get(
                cell,
                [],
            )
        )

    # ==================================================================
    # ROUTE 2 — WORD GRID
    # ==================================================================

    def _word_cell(
        self,
        token_info: Dict[str, Any],
    ) -> tuple[int, int]:
        """
        Return the A×A Word Grid coordinate supplied by tokenizer.py.

        Established Word Grid rule:

            L   = word/name length
            uID = L
            S   = L

        tokenizer.py resolves that information into:

            word["row"]
            word["col"]

        MemoryGrid does not replace that resolution with the 64-row
        storage formula.

        This separation is intentional.
        """

        word = token_info.get(
            "word"
        ) or {}

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

        return row, col

    def _place_word(
        self,
        token_info: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """
        Place the complete word into its A×A Word Grid cell.
        """

        row, col = self._word_cell(
            token_info
        )

        cell = (
            row,
            col,
        )

        if cell not in self.word_grid:
            self.word_grid[cell] = []

        self.word_grid[cell].append(
            record
        )

    def get_tokens_at_word(
        self,
        row: int,
        col: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve token records from an A×A Word Grid cell.
        """

        cell = (
            int(row),
            int(col),
        )

        return list(
            self.word_grid.get(
                cell,
                [],
            )
        )

    # ==================================================================
    # ROUTE 3 — FULL-TEXT / STORAGE GRID
    # ==================================================================

    def _storage_cells(
        self,
        token_info: Dict[str, Any],
    ) -> List[int]:
        """
        Resolve the independent 64-row storage position.

        The storage route is based on the length of the word contained
        in the indexed entry.

        IMPORTANT:

            This is NOT the Word Grid.

            Word Grid:
                A × A
                L = name length
                uID = L
                S = L

            Storage:
                independent 64-row dimension

        The storage row is therefore calculated independently from the
        tokenizer's A×A word coordinate.
        """

        word = token_info.get(
            "original",
            "",
        )

        if not word:
            return []

        length = len(
            word
        )

        row = (
            length
            % self.rows
        )

        return [row]

    def _place_storage(
        self,
        token_info: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """
        Place a token into the independent storage route.
        """

        storage_rows = self._storage_cells(
            token_info
        )

        for row in storage_rows:

            if row not in self.storage_grid:
                self.storage_grid[row] = []

            self.storage_grid[row].append(
                record
            )

    def get_tokens_at_storage(
        self,
        row: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve token records from the independent 64-row storage
        dimension.
        """

        normalized_row = (
            int(row)
            % self.rows
        )

        return list(
            self.storage_grid.get(
                normalized_row,
                [],
            )
        )

    # ==================================================================
    # DOCUMENT INSERTION
    # ==================================================================

    def add_document(
        self,
        text: str,
        lang: str = "en",
        source: str = "",
    ) -> int:
        """
        Add an entry to CoMpaNeoN memory.

        The source is intentionally generic.

        Therefore this method can index:

            user entries
            crawled content
            background-training material
            dictionary knowledge
            research
            domain knowledge
            other indexed sources

        Every token enters the three independent routes:

            1. Letter Grid
            2. Word Grid
            3. 64-row Storage Grid

        The routes are not collapsed into one coordinate system.
        """

        lang = normalize_lang(
            lang
        )

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
                "original": token.get(
                    "original",
                    "",
                ),
                "stem": token.get(
                    "stem",
                    "",
                ),
                "word": token.get(
                    "word",
                    {},
                ),
                "letter": token.get(
                    "letter",
                    [],
                ),
                "lang": token.get(
                    "lang",
                    lang,
                ),
                "source": source,
            }

            # ----------------------------------------------------------
            # ROUTE 1 — LETTER GRID
            # ----------------------------------------------------------

            for letter_index in self._letter_cells(
                token
            ):
                self._place_letter(
                    letter_index,
                    record,
                )

            # ----------------------------------------------------------
            # ROUTE 2 — WORD GRID
            # ----------------------------------------------------------

            self._place_word(
                token,
                record,
            )

            # ----------------------------------------------------------
            # ROUTE 3 — STORAGE GRID
            # ----------------------------------------------------------

            self._place_storage(
                token,
                record,
            )

        return doc_id

    # ==================================================================
    # RETRIEVAL — ONE TOKEN
    # ==================================================================

    def retrieve_by_token(
        self,
        token_info: Dict[str, Any],
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:
        """
        Retrieve candidates for one token through all three routes.

        Retrieval remains separated by route.

        No ranking is performed here.

        Returns:

            {
                "letter": [...],
                "word": [...],
                "storage": [...]
            }
        """

        result = {
            "letter": [],
            "word": [],
            "storage": [],
        }

        # --------------------------------------------------------------
        # LETTER ROUTE
        # --------------------------------------------------------------

        seen_letter = set()

        for letter_index in self._letter_cells(
            token_info
        ):

            entries = self.get_tokens_at_letter(
                letter_index
            )

            for item in entries:

                key = (
                    item.get("doc_id"),
                    item.get("original"),
                )

                if key in seen_letter:
                    continue

                seen_letter.add(
                    key
                )

                result["letter"].append(
                    item
                )

        # --------------------------------------------------------------
        # WORD ROUTE
        # --------------------------------------------------------------

        word_row, word_col = self._word_cell(
            token_info
        )

        word_entries = self.get_tokens_at_word(
            word_row,
            word_col,
        )

        seen_word = set()

        for item in word_entries:

            key = (
                item.get("doc_id"),
                item.get("original"),
            )

            if key in seen_word:
                continue

            seen_word.add(
                key
            )

            result["word"].append(
                item
            )

        # --------------------------------------------------------------
        # STORAGE ROUTE
        # --------------------------------------------------------------

        seen_storage = set()

        for storage_row in self._storage_cells(
            token_info
        ):

            storage_entries = (
                self.get_tokens_at_storage(
                    storage_row
                )
            )

            for item in storage_entries:

                key = (
                    item.get("doc_id"),
                    item.get("original"),
                )

                if key in seen_storage:
                    continue

                seen_storage.add(
                    key
                )

                result["storage"].append(
                    item
                )

        return result

    # ==================================================================
    # RETRIEVAL — MULTIPLE TOKENS
    # ==================================================================

    def retrieve(
        self,
        token_infos: List[
            Dict[str, Any]
        ],
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:
        """
        Retrieve candidates for multiple query tokens.

        The three routes remain separate.

        This method does not decide which route is more important.
        Ranking belongs to the higher-level understanding/ranking
        architecture.
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

            for route in (
                "letter",
                "word",
                "storage",
            ):

                for item in routes[route]:

                    key = (
                        item.get("doc_id"),
                        item.get("original"),
                    )

                    if key in seen[route]:
                        continue

                    seen[route].add(
                        key
                    )

                    result[route].append(
                        item
                    )

        return result

    # ==================================================================
    # DOCUMENT ACCESS
    # ==================================================================

    def get_doc(
        self,
        doc_id: int,
    ) -> str:
        """
        Return complete document text.
        """

        if not (
            0 <= doc_id < len(
                self.doc_store
            )
        ):
            return ""

        return self.doc_store[
            doc_id
        ].get(
            "text",
            "",
        )

    def get_doc_tokens(
        self,
        doc_id: int,
    ) -> list:
        """
        Return tokenizer output for a document.
        """

        if not (
            0 <= doc_id < len(
                self.doc_store
            )
        ):
            return []

        return self.doc_store[
            doc_id
        ].get(
            "tokens",
            [],
        )

    def get_doc_source(
        self,
        doc_id: int,
    ) -> str:
        """
        Return the source associated with a document.
        """

        if not (
            0 <= doc_id < len(
                self.doc_store
            )
        ):
            return ""

        return self.doc_store[
            doc_id
        ].get(
            "source",
            "",
        )

    def get_doc_language(
        self,
        doc_id: int,
    ) -> str:
        """
        Return the normalized language of a document.
        """

        if not (
            0 <= doc_id < len(
                self.doc_store
            )
        ):
            return ""

        return self.doc_store[
            doc_id
        ].get(
            "lang",
            "en",
        )

    # ==================================================================
    # LEGACY COMPATIBILITY
    # ==================================================================

    def get_tokens_at(
        self,
        row: int,
        col: int,
    ) -> List[Dict[str, Any]]:
        """
        Legacy compatibility accessor.

        Historically callers used:

            get_tokens_at(row, col)

        Under the three-route architecture, this is interpreted as
        Word Grid access.

        New code should use the explicit methods:

            get_tokens_at_letter()
            get_tokens_at_word()
            get_tokens_at_storage()
        """

        return self.get_tokens_at_word(
            row,
            col,
        )