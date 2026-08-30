"""
CoMpaNeoN Memory Grid
=====================

Knowledge Memory Grid for CoMpaNeoN.

Architecture
------------

MemoryGrid maintains three independent memory routes:

    ROUTE 1 — LETTER GRID
        Language alphabet × 1

    ROUTE 2 — WORD GRID
        Language alphabet × alphabet

    ROUTE 3 — FULL-TEXT / STORAGE GRID
        Independent 64-row GSP storage dimension

Responsibilities
----------------

MemoryGrid:

    - stores tokenized knowledge
    - stores complete documents
    - indexes letters
    - indexes complete words
    - places complete text into the independent GSP storage grid
    - retrieves candidates through independent routes

MemoryGrid does NOT:

    - define alphabets
    - tokenize text
    - calculate letter indices
    - calculate word-grid coordinates
    - perform crawler perturbation
    - perform GSP K replication
    - perform GSP D movement
    - rank results
    - generate prompts
    - generate AI responses

Dependencies
------------

tokenizer.py
    Owns linguistic representation:

        tokenize()
        normalize_lang()

    Token records provide:

        letter
        word
        stem
        language

keyboard.py
    Owns full-text GSP calculation:

        calculate_lsum()
        calculate_ssum()
        first_letter_index()
        gsp_place()

    MemoryGrid uses only the GSP start-row result.

IMPORTANT GSP RULE
------------------

For full-text storage:

    L = calculate_lsum(text)
    S = calculate_ssum(text)
    c = first_letter_index(text)
    R = 64

    start_row = ((L + S - 1) % R) + 1

The value of c is preserved directly.

MemoryGrid does NOT calculate:

    c % 26

K and D
-------

K and D do NOT belong to MemoryGrid placement.

They belong to crawler/grid-crawler movement and perturbation.

Therefore MemoryGrid performs exactly one GSP storage placement:

    full text -> start_row

No replica sequence is generated here.
No forward D is generated here.
No backward D is generated here.

The three routes are unified only during retrieval/ranking.
"""


from __future__ import annotations


from typing import (
    List,
    Dict,
    Any,
)


# =====================================================================
# TOKENIZER
# =====================================================================

from tokenizer import (
    tokenize,
    normalize_lang,
)


# =====================================================================
# GSP / KEYBOARD
# =====================================================================

from keyboard import (
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    gsp_place,
)


# =====================================================================
# MEMORY GRID
# =====================================================================

class MemoryGrid:
    """
    CoMpaNeoN Knowledge Memory Grid.

    Knowledge may come from:

        - user entries
        - crawled documents
        - background training
        - dictionary/domain knowledge
        - research material
        - indexed project knowledge
        - other approved knowledge sources

    Every entry is represented through three independent routes:

        1. Letter Grid
        2. Word Grid
        3. Full-text / Storage Grid

    The routes remain independent.

    Tokenizer owns linguistic coordinates.

    Keyboard/GSP owns full-text storage placement.

    MemoryGrid owns storage and retrieval.
    """

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(
        self,
        rows: int = 64,
        cols: int = 26,
    ):
        """
        Initialize the memory grid.

        rows:
            Independent full-text storage dimension.

            Default:
                64

        cols:
            Default alphabet/column dimension.

            This remains available for compatibility with the
            established English 26-column architecture.

        Language-specific alphabet dimensions remain owned by
        tokenizer.py.
        """

        self.rows = int(rows)

        self.cols = int(cols)

        # -------------------------------------------------------------
        # ROUTE 1
        # LETTER GRID
        #
        # alphabet index -> token records
        # -------------------------------------------------------------

        self.letter_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # ROUTE 2
        # WORD GRID
        #
        # (word_row, word_col) -> token records
        #
        # Coordinates are supplied by tokenizer.py.
        # -------------------------------------------------------------

        self.word_grid: Dict[
            tuple[int, int],
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # ROUTE 3
        # FULL-TEXT / STORAGE GRID
        #
        # GSP start row -> document/token records
        #
        # Independent 64-row storage dimension.
        # -------------------------------------------------------------

        self.storage_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # COMPLETE DOCUMENT STORE
        #
        # Preserves complete original text.
        # -------------------------------------------------------------

        self.doc_store: List[
            Dict[str, Any]
        ] = []

    # =================================================================
    # ROUTE 1 — LETTER GRID
    # =================================================================

    def _letter_cells(
        self,
        token_info: Dict[str, Any],
    ) -> List[int]:
        """
        Return the letter-grid positions belonging to a token.

        tokenizer.py has already calculated the alphabet indices.

        MemoryGrid does not recalculate them.

        No storage row calculation occurs here.
        No GSP calculation occurs here.
        """

        cells = token_info.get(
            "letter",
            [],
        )

        if not cells:
            return []

        return [
            int(cell)
            for cell in cells
        ]

    # -----------------------------------------------------------------

    def _place_letter(
        self,
        letter_index: int,
        record: Dict[str, Any],
    ) -> None:
        """
        Place a token record into one Letter Grid cell.
        """

        cell = int(
            letter_index
        )

        if cell not in self.letter_grid:

            self.letter_grid[cell] = []

        self.letter_grid[cell].append(
            record
        )

    # -----------------------------------------------------------------

    def get_tokens_at_letter(
        self,
        letter_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve token records from a Letter Grid position.
        """

        cell = int(
            letter_index
        )

        return list(
            self.letter_grid.get(
                cell,
                [],
            )
        )

    # =================================================================
    # ROUTE 2 — WORD GRID
    # =================================================================

    def _word_cell(
        self,
        token_info: Dict[str, Any],
    ) -> tuple[int, int]:
        """
        Return the A×A Word Grid coordinate supplied by tokenizer.py.

        tokenizer.py owns:

            L
            uID
            word_S
            c
            col
            row
            A

        MemoryGrid simply consumes:

            word["row"]
            word["col"]

        No 64-row storage calculation occurs here.
        No K occurs here.
        No D occurs here.
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

        return (
            row,
            col,
        )

    # -----------------------------------------------------------------

    def _place_word(
        self,
        token_info: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        """
        Place the complete word into its tokenizer-defined
        A×A Word Grid cell.
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

    # -----------------------------------------------------------------

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

    # =================================================================
    # ROUTE 3 — FULL-TEXT / STORAGE GRID
    # =================================================================

    def _storage_gsp(
        self,
        text: str,
        lang: str = "en",
    ) -> Dict[str, int]:
        """
        Calculate the full-text GSP storage coordinates.

        keyboard.py owns the GSP calculation.

        This method intentionally does NOT reproduce the formulas.

        It calls:

            calculate_lsum()
            calculate_ssum()
            first_letter_index()
            gsp_place()

        The resulting start row is the storage coordinate.

        IMPORTANT:

            c is preserved directly.

            No:

                c % 26

            is performed.

        IMPORTANT:

            K is not supplied.

            D is not supplied.

            C is not supplied.

            MemoryGrid needs only the GSP start row for storage.

        The canonical row dimension is:

            R = self.rows

        Default:

            R = 64
        """

        if not text:

            return {
                "L": 0,
                "S": 0,
                "c": 0,
                "R": self.rows,
                "start_row": 0,
            }

        # -------------------------------------------------------------
        # Full-text GSP inputs
        # -------------------------------------------------------------

        Lsum = calculate_lsum(
            text,
            lang,
        )

        Ssum = calculate_ssum(
            text,
            lang,
        )

        c = first_letter_index(
            text,
            lang,
        )

        # -------------------------------------------------------------
        # Main GSP placement
        #
        # keyboard.py owns:
        #
        #     start_row = ((L + S - 1) % R) + 1
        #
        # We deliberately do not reproduce it here.
        # -------------------------------------------------------------

        placement = gsp_place(
            Lsum,
            Ssum,
            c,
            R=self.rows,
        )

        return {
            "L": int(Lsum),
            "S": int(Ssum),
            "c": int(c),
            "R": self.rows,
            "start_row": int(
                placement.get(
                    "start_row",
                    0,
                )
            ),
        }

    # -----------------------------------------------------------------

    def _storage_cells(
        self,
        text: str,
        lang: str = "en",
    ) -> List[int]:
        """
        Resolve the independent full-text storage cell.

        Exactly one primary GSP start row is returned.

        There is no K sequence.

        There is no D movement.

        There is no forward perturbation.

        There is no backward perturbation.
        """

        if not text:

            return []

        gsp = self._storage_gsp(
            text,
            lang,
        )

        start_row = int(
            gsp.get(
                "start_row",
                0,
            )
        )

        if start_row <= 0:

            return []

        return [
            start_row
        ]

    # -----------------------------------------------------------------

    def _place_storage(
        self,
        text: str,
        lang: str,
        record: Dict[str, Any],
    ) -> None:
        """
        Place complete-text knowledge into the independent
        64-row GSP storage grid.
        """

        storage_rows = self._storage_cells(
            text,
            lang,
        )

        for row in storage_rows:

            if row not in self.storage_grid:

                self.storage_grid[row] = []

            self.storage_grid[row].append(
                record
            )

    # -----------------------------------------------------------------

    def get_tokens_at_storage(
        self,
        row: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve records from the independent GSP storage row.

        Storage rows are normalized against the configured storage
        dimension for compatibility.

        The stored GSP start row itself remains 1-based.
        """

        normalized_row = int(row)

        if normalized_row <= 0:

            return []

        if normalized_row > self.rows:

            normalized_row = (
                ((normalized_row - 1) % self.rows)
                + 1
            )

        return list(
            self.storage_grid.get(
                normalized_row,
                [],
            )
        )

    # =================================================================
    # DOCUMENT INSERTION
    # =================================================================

    def add_document(
        self,
        text: str,
        lang: str = "en",
        source: str = "",
    ) -> int:
        """
        Add an entry to CoMpaNeoN memory.

        The text is first passed through tokenizer.py.

        Every token enters:

            1. Letter Grid
            2. Word Grid

        The complete original document enters:

            3. Full-text GSP Storage Grid

        This distinction is important.

        Token-level linguistic placement and complete-text storage
        placement are separate operations.

        GSP storage placement uses the complete text.

        It does not calculate a storage row independently for every
        token.
        """

        lang = normalize_lang(
            lang
        )

        # -------------------------------------------------------------
        # TOKENIZATION
        # -------------------------------------------------------------

        tokens = tokenize(
            text,
            lang,
        )

        # -------------------------------------------------------------
        # DOCUMENT ID
        # -------------------------------------------------------------

        doc_id = len(
            self.doc_store
        )

        # -------------------------------------------------------------
        # DOCUMENT STORE
        #
        # Preserve complete original knowledge.
        # -------------------------------------------------------------

        self.doc_store.append({

            "text": text,

            "source": source,

            "lang": lang,

            "tokens": tokens,

        })

        # -------------------------------------------------------------
        # TOKEN ROUTES
        # -------------------------------------------------------------

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

            # ---------------------------------------------------------
            # ROUTE 1 — LETTER GRID
            # ---------------------------------------------------------

            for letter_idx in self._letter_cells(
                token
            ):

                self._place_letter(
                    letter_idx,
                    record,
                )

            # ---------------------------------------------------------
            # ROUTE 2 — WORD GRID
            # ---------------------------------------------------------

            self._place_word(
                token,
                record,
            )

        # -------------------------------------------------------------
        # ROUTE 3 — FULL-TEXT GSP STORAGE
        #
        # IMPORTANT:
        #
        # This is performed once for the complete entry.
        #
        # K and D do not participate.
        # -------------------------------------------------------------

        document_record = {

            "doc_id": doc_id,

            "original": text,

            "stem": "",

            "word": {},

            "letter": [],

            "lang": lang,

            "source": source,

            "storage_gsp": self._storage_gsp(
                text,
                lang,
            ),

        }

        self._place_storage(
            text,
            lang,
            document_record,
        )

        return doc_id

    # =================================================================
    # RETRIEVAL — ONE TOKEN
    # =================================================================

    def retrieve_by_token(
        self,
        token_info: Dict[str, Any],
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:
        """
        Retrieve candidates for one token through the independent
        token routes.

        Returns:

            {
                "letter": [...],
                "word": [...],
                "storage": [...]
            }

        No ranking occurs here.
        """

        result = {

            "letter": [],

            "word": [],

            "storage": [],

        }

        # -------------------------------------------------------------
        # LETTER ROUTE
        # -------------------------------------------------------------

        seen_letter = set()

        for letter_idx in self._letter_cells(
            token_info
        ):

            entries = self.get_tokens_at_letter(
                letter_idx
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

        # -------------------------------------------------------------
        # WORD ROUTE
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # STORAGE ROUTE
        #
        # Token retrieval can use the complete original document
        # associated with the token.
        #
        # We therefore resolve the token's document and use the
        # document's complete text for the GSP storage lookup.
        # -------------------------------------------------------------

        doc_id = token_info.get(
            "doc_id"
        )

        if doc_id is not None:

            try:

                document = self.doc_store[
                    int(doc_id)
                ]

            except (
                IndexError,
                ValueError,
                TypeError,
            ):

                document = None

            if document:

                document_text = document.get(
                    "text",
                    "",
                )

                document_lang = document.get(
                    "lang",
                    "en",
                )

                storage_rows = self._storage_cells(
                    document_text,
                    document_lang,
                )

                seen_storage = set()

                for storage_row in storage_rows:

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

    # =================================================================
    # RETRIEVAL — MULTIPLE TOKENS
    # =================================================================

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

        The routes remain independent.

        Ranking belongs to the higher-level ranking/understanding
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

    # =================================================================
    # DIRECT DOCUMENT STORAGE LOOKUP
    # =================================================================

    def retrieve_document_by_gsp(
        self,
        text: str,
        lang: str = "en",
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Retrieve complete-text storage candidates using the text's
        GSP start row.

        This is the direct full-text/storage lookup.

        It uses:

            keyboard.py
                calculate_lsum()
                calculate_ssum()
                first_letter_index()
                gsp_place()

        No K.
        No D.
        No perturbation.
        """

        lang = normalize_lang(
            lang
        )

        storage_rows = self._storage_cells(
            text,
            lang,
        )

        result = []

        seen = set()

        for row in storage_rows:

            entries = self.get_tokens_at_storage(
                row
            )

            for item in entries:

                key = (
                    item.get("doc_id"),
                    item.get("original"),
                )

                if key in seen:

                    continue

                seen.add(
                    key
                )

                result.append(
                    item
                )

        return result

    # =================================================================
    # DOCUMENT ACCESS
    # =================================================================

    def get_doc(
        self,
        doc_id: int,
    ) -> str:
        """
        Return complete original document text.
        """

        try:

            index = int(
                doc_id
            )

        except (
            ValueError,
            TypeError,
        ):

            return ""

        if not (
            0 <= index < len(
                self.doc_store
            )
        ):

            return ""

        return self.doc_store[
            index
        ].get(
            "text",
            "",
        )

    # -----------------------------------------------------------------

    def get_doc_tokens(
        self,
        doc_id: int,
    ) -> list:
        """
        Return tokenizer output for a document.
        """

        try:

            index = int(
                doc_id
            )

        except (
            ValueError,
            TypeError,
        ):

            return []

        if not (
            0 <= index < len(
                self.doc_store
            )
        ):

            return []

        return self.doc_store[
            index
        ].get(
            "tokens",
            [],
        )

    # -----------------------------------------------------------------

    def get_doc_source(
        self,
        doc_id: int,
    ) -> str:
        """
        Return the source associated with a document.
        """

        try:

            index = int(
                doc_id
            )

        except (
            ValueError,
            TypeError,
        ):

            return ""

        if not (
            0 <= index < len(
                self.doc_store
            )
        ):

            return ""

        return self.doc_store[
            index
        ].get(
            "source",
            "",
        )

    # -----------------------------------------------------------------

    def get_doc_language(
        self,
        doc_id: int,
    ) -> str:
        """
        Return the normalized language of a document.
        """

        try:

            index = int(
                doc_id
            )

        except (
            ValueError,
            TypeError,
        ):

            return ""

        if not (
            0 <= index < len(
                self.doc_store
            )
        ):

            return ""

        return self.doc_store[
            index
        ].get(
            "lang",
            "en",
        )

    # =================================================================
    # GSP DOCUMENT INFORMATION
    # =================================================================

    def get_doc_gsp(
        self,
        doc_id: int,
    ) -> Dict[str, int]:
        """
        Return the canonical full-text GSP information for a document.

        The returned structure contains:

            L
            S
            c
            R
            start_row

        This is useful for inspection and diagnostics.
        """

        try:

            index = int(
                doc_id
            )

        except (
            ValueError,
            TypeError,
        ):

            return {}

        if not (
            0 <= index < len(
                self.doc_store
            )
        ):

            return {}

        document = self.doc_store[
            index
        ]

        text = document.get(
            "text",
            "",
        )

        lang = document.get(
            "lang",
            "en",
        )

        return self._storage_gsp(
            text,
            lang,
        )

    # =================================================================
    # GRID INSPECTION
    # =================================================================

    def grid_stats(
        self,
    ) -> Dict[str, int]:
        """
        Return basic MemoryGrid statistics.
        """

        return {

            "documents": len(
                self.doc_store
            ),

            "letter_cells": len(
                self.letter_grid
            ),

            "word_cells": len(
                self.word_grid
            ),

            "storage_rows": len(
                self.storage_grid
            ),

            "storage_dimension": self.rows,

        }

    # =================================================================
    # LEGACY COMPATIBILITY
    # =================================================================

    def get_tokens_at(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Legacy compatibility accessor.

        Historically callers used:

            get_tokens_at(row, col)

        Under the current three-route architecture this continues
        to mean Word Grid access.

        New code should prefer:

            get_tokens_at_letter()
            get_tokens_at_word()
            get_tokens_at_storage()
        """

        return self.get_tokens_at_word(
            row,
            col,
        )


# =====================================================================
# DEVELOPMENT TEST
# =====================================================================

if __name__ == "__main__":

    grid = MemoryGrid()

    sample_text = (
        "CoMpaNeoN deterministic "
        "knowledge memory grid"
    )

    print(
        "Adding document..."
    )

    doc_id = grid.add_document(
        sample_text,
        "en",
        source="development",
    )

    print(
        "\nDocument ID:"
    )

    print(
        doc_id
    )

    print(
        "\nDocument:"
    )

    print(
        grid.get_doc(
            doc_id
        )
    )

    print(
        "\nDocument GSP:"
    )

    print(
        grid.get_doc_gsp(
            doc_id
        )
    )

    print(
        "\nGrid statistics:"
    )

    print(
        grid.grid_stats()
    )

    print(
        "\nDocument tokens:"
    )

    for token in grid.get_doc_tokens(
        doc_id
    ):

        print(
            token
        )
