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
