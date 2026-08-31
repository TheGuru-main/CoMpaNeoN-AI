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
    - generate crawler perturbation
    - generate GSP replica sequences
    - rank results
    - generate prompts
    - generate AI responses


Placement Architecture
----------------------

tokenizer.py
    owns linguistic token representation

placement.py
    owns placement mode and placement identity

keyboard.py
    owns keyboard-derived L/S/c calculations
    where applicable

MemoryGrid
    consumes placement results for storage.


IMPORTANT
---------

For full-text storage:

    placement.py determines:

        L
        S
        c
        start_row

The randomized UID-derived S used for full-text storage
is preserved through placement.py.

MemoryGrid does not generate that UID.

K and D remain outside MemoryGrid.

They belong to crawler/grid traversal and replication logic.
"""


from __future__ import annotations


from typing import (
    List,
    Dict,
    Any,
    Optional,
)


# =====================================================================
# TOKENIZER
# =====================================================================

from tokenizer import (
    tokenize,
    normalize_lang,
)


# =====================================================================
# PLACEMENT
# =====================================================================

from placement import (
    place_full_text,
)


# =====================================================================
# MEMORY GRID
# =====================================================================

class MemoryGrid:
    """
    CoMpaNeoN Knowledge Memory Grid.

    Knowledge may originate from:

        - user entries
        - crawled documents
        - background training
        - trained datasets
        - dictionary knowledge
        - domain knowledge
        - research material
        - indexed project knowledge
        - AI interaction
        - approved external sources

    Every entry is represented through three independent routes:

        1. Letter Grid
        2. Word Grid
        3. Full-text / Storage Grid

    The routes remain independent.

    Tokenizer owns linguistic coordinates.

    Placement owns identity-aware placement decisions.

    MemoryGrid owns storage and route retrieval.
    """

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(
        self,
        rows: int = 64,
        cols: int = 26,
    ) -> None:

        self.rows = int(rows)
        self.cols = int(cols)

        # -------------------------------------------------------------
        # ROUTE 1 — LETTER GRID
        # -------------------------------------------------------------

        self.letter_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # ROUTE 2 — WORD GRID
        # -------------------------------------------------------------

        self.word_grid: Dict[
            tuple[int, int],
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # ROUTE 3 — FULL-TEXT STORAGE GRID
        #
        # start_row -> complete document records
        # -------------------------------------------------------------

        self.storage_grid: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # -------------------------------------------------------------
        # COMPLETE DOCUMENT STORE
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

        cell = int(letter_index)

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

        return list(
            self.letter_grid.get(
                int(letter_index),
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

        return list(
            self.word_grid.get(
                (
                    int(row),
                    int(col),
                ),
                [],
            )
        )

    # =================================================================
    # ROUTE 3 — FULL-TEXT STORAGE GRID
    # =================================================================

    def _storage_gsp(
        self,
        text: str,
        lang: str = "en",
        storage_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve canonical full-text placement.

        placement.py owns:

            - full-text placement mode
            - UID-derived S
            - L calculation routing
            - c preservation
            - start-row calculation routing

        MemoryGrid consumes the result.

        storage_uid:

            Optional externally supplied randomized UID.

            If omitted, placement.py is responsible for generating
            the randomized storage identity.
        """

        if not text:

            return {
                "L": 0,
                "S": 0,
                "c": 0,
                "R": self.rows,
                "start_row": 0,
            }

        placement = place_full_text(
            text=text,
            lang=lang,
            storage_uid=storage_uid,
            rows=self.rows,
        )

        return {
            "L": int(
                placement.get(
                    "L",
                    0,
                )
            ),

            "S": int(
                placement.get(
                    "S",
                    0,
                )
            ),

            "c": int(
                placement.get(
                    "c",
                    0,
                )
            ),

            "R": int(
                placement.get(
                    "R",
                    self.rows,
                )
            ),

            "start_row": int(
                placement.get(
                    "start_row",
                    0,
                )
            ),

            "storage_uid": placement.get(
                "storage_uid",
                "",
            ),
        }

    # -----------------------------------------------------------------

    def _storage_cells(
        self,
        text: str,
        lang: str = "en",
        storage_uid: Optional[str] = None,
    ) -> List[int]:

        if not text:
            return []

        gsp = self._storage_gsp(
            text=text,
            lang=lang,
            storage_uid=storage_uid,
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
        storage_uid: Optional[str] = None,
    ) -> None:

        storage_rows = self._storage_cells(
            text=text,
            lang=lang,
            storage_uid=storage_uid,
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

        normalized_row = int(row)

        if normalized_row <= 0:
            return []

        if normalized_row > self.rows:

            normalized_row = (
                (
                    normalized_row - 1
                )
                % self.rows
            ) + 1

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
        storage_uid: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> int:

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

        metadata = metadata or {}

        # -------------------------------------------------------------
        # Resolve placement once.
        #
        # Important:
        #
        # The same placement identity must be stored with the document.
        #
        # This prevents retrieval from generating a new random UID.
        # -------------------------------------------------------------

        storage_gsp = self._storage_gsp(
            text=text,
            lang=lang,
            storage_uid=storage_uid,
        )

        resolved_storage_uid = (
            storage_gsp.get(
                "storage_uid",
                ""
            )
        )

        # -------------------------------------------------------------
        # DOCUMENT STORE
        # -------------------------------------------------------------

        self.doc_store.append({

            "doc_id": doc_id,

            "text": text,

            "source": source,

            "lang": lang,

            "tokens": tokens,

            "storage_uid": (
                resolved_storage_uid
            ),

            "storage_gsp": (
                storage_gsp
            ),

            "metadata": metadata,

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
            # LETTER ROUTE
            # ---------------------------------------------------------

            for letter_idx in self._letter_cells(
                token
            ):

                self._place_letter(
                    letter_idx,
                    record,
                )

            # ---------------------------------------------------------
            # WORD ROUTE
            # ---------------------------------------------------------

            self._place_word(
                token,
                record,
            )

        # -------------------------------------------------------------
        # FULL-TEXT STORAGE ROUTE
        # -------------------------------------------------------------

        document_record = {

            "doc_id": doc_id,

            "original": text,

            "lang": lang,

            "source": source,

            "storage_uid": (
                resolved_storage_uid
            ),

            "storage_gsp": (
                storage_gsp
            ),

            "metadata": metadata,

        }

        start_row = int(
            storage_gsp.get(
                "start_row",
                0,
            )
        )

        if start_row > 0:

            if start_row not in self.storage_grid:

                self.storage_grid[
                    start_row
                ] = []

            self.storage_grid[
                start_row
            ].append(
                document_record
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

        result = {

            "letter": [],

            "word": [],

            "storage": [],

        }

        # -------------------------------------------------------------
        # LETTER ROUTE
        # -------------------------------------------------------------

        seen_letter = set()

        for letter_index in self._letter_cells(
            token_info
        ):

            entries = self.get_tokens_at_letter(
                letter_index
            )

            for item in entries:

                key = (
                    item.get(
                        "doc_id"
                    ),

                    item.get(
                        "original"
                    ),
                )

                if key in seen_letter:
                    continue

                seen_letter.add(
                    key
                )

                result[
                    "letter"
                ].append(
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
                item.get(
                    "doc_id"
                ),

                item.get(
                    "original"
                ),
            )

            if key in seen_word:
                continue

            seen_word.add(
                key
            )

            result[
                "word"
            ].append(
                item
            )

        # -------------------------------------------------------------
        # STORAGE ROUTE
        #
        # Token may carry its originating doc_id.
        #
        # We use the stored document placement instead of generating
        # a new randomized full-text UID.
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

                storage_gsp = (
                    document.get(
                        "storage_gsp",
                        {}
                    )
                )

                storage_row = int(
                    storage_gsp.get(
                        "start_row",
                        0,
                    )
                )

                if storage_row > 0:

                    seen_storage = set()

                    entries = (
                        self.get_tokens_at_storage(
                            storage_row
                        )
                    )

                    for item in entries:

                        key = (
                            item.get(
                                "doc_id"
                            ),

                            item.get(
                                "original"
                            ),
                        )

                        if key in seen_storage:
                            continue

                        seen_storage.add(
                            key
                        )

                        result[
                            "storage"
                        ].append(
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

                for item in routes[
                    route
                ]:

                    key = (

                        item.get(
                            "doc_id"
                        ),

                        item.get(
                            "original"
                        ),
                    )

                    if key in seen[
                        route
                    ]:
                        continue

                    seen[
                        route
                    ].add(
                        key
                    )

                    result[
                        route
                    ].append(
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
        storage_uid: Optional[str] = None,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Direct lookup by canonical placement identity.

        If a randomized storage UID is supplied,
        the lookup can reproduce that document's placement.

        Without the original storage UID, a randomized full-text
        placement cannot reliably reproduce the original cell.

        Broad discovery of unknown full-text entries belongs to
        GridCrawler.
        """

        lang = normalize_lang(
            lang
        )

        if not storage_uid:

            return []

        gsp = self._storage_gsp(
            text=text,
            lang=lang,
            storage_uid=storage_uid,
        )

        row = int(
            gsp.get(
                "start_row",
                0,
            )
        )

        if row <= 0:
            return []

        return self.get_tokens_at_storage(
            row
        )

    # =================================================================
    # DOCUMENT ACCESS
    # =================================================================

    def get_doc(
        self,
        doc_id: int,
    ) -> str:

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
            0 <= index
            < len(
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
    ) -> List[
        Dict[str, Any]
    ]:

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
            0 <= index
            < len(
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
            0 <= index
            < len(
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
            0 <= index
            < len(
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
    ) -> Dict[str, Any]:

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
            0 <= index
            < len(
                self.doc_store
            )
        ):

            return {}

        return dict(
            self.doc_store[
                index
            ].get(
                "storage_gsp",
                {}
            )
        )

    # =================================================================
    # GRID INSPECTION
    # =================================================================

    def grid_stats(
        self,
    ) -> Dict[str, int]:

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

    doc_id = grid.add_document(
        text=sample_text,
        lang="en",
        source="development",
    )

    print(
        "Document ID:",
        doc_id,
    )

    print(
        "Document:",
        grid.get_doc(
            doc_id
        ),
    )

    print(
        "Document GSP:",
        grid.get_doc_gsp(
            doc_id
        ),
    )

    print(
        "Grid statistics:",
        grid.grid_stats(),
    )