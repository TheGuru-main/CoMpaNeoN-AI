"""
CoMpaNeoN Memory Grid
=====================

Canonical multilingual spatial memory for CoMpaNeoN.

ARCHITECTURE
------------

External / User / AI Text
        │
        ▼
Language Detection
        │
        ▼
tokenizer.py
        │
        ├── normalization
        ├── language switching
        ├── linguistic phase
        ├── alphabet / symbol mapping
        └── tokenization
        │
        ▼
keyboard.py / placement
        │
        ▼
MemoryGrid
46 Columns × 64 Rows
        │
        ├── documents
        ├── tokens
        ├── storage rows
        ├── word cells
        ├── language phases
        └── document metadata
        │
        ▼
Crawler Rooters
        │
        ├── GridCrawler
        ├── WebCrawler
        ├── other crawlers
        ├── CrawlerScheduler
        └── CrawlerRetrieval


ARCHITECTURAL AUTHORITY
-----------------------

tokenizer.py
    Owns:
        - language normalization
        - language switching
        - linguistic representation
        - tokenization
        - multilingual phase information

keyboard.py / placement
    Owns:
        - canonical GSP mathematics
        - Lsum
        - Ssum
        - first-letter / linguistic entry mapping
        - canonical placement mathematics

MemoryGrid
    Owns:
        - 46 × 64 multilingual memory space
        - document storage
        - token storage
        - metadata preservation
        - content identity
        - storage lookup
        - word-grid lookup
        - crawler integration point

GridCrawler
    Owns:
        - K = 250
        - forward_d = 5
        - backward_k = 5
        - backward_d = 1
        - forward traversal
        - backward perturbations
        - Elastic Cloud traversal
        - rooting through MemoryGrid

WebCrawler
    Owns:
        - external acquisition
        - HTML/API/video acquisition
        - transcript acquisition
        - external content ingestion

CrawlerScheduler
    Owns:
        - crawler scheduling
        - crawler execution coordination

CrawlerRetrieval
    Owns:
        - retrieval orchestration
        - crawler candidate coordination

IMPORTANT
---------

MemoryGrid does NOT own crawler traversal mathematics.

K and D do NOT belong to MemoryGrid.

The crawlers are the rooters.

The canonical multilingual grid is:

    46 columns
    64 rows

All crawler column traversal must therefore use:

    memory_grid.cols

and must never assume 26 columns.
"""

from __future__ import annotations

import hashlib

from datetime import datetime, timezone

from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# OPTIONAL LANGUAGE DETECTION
# ============================================================================

try:

    from langdetect import detect as detect_language_code

except ImportError:

    detect_language_code = None


# ============================================================================
# TOKENIZER
# ============================================================================

from tokenizer import (
    normalize_lang,
    tokenize,
)


# ============================================================================
# KEYBOARD / PLACEMENT
# ============================================================================

from keyboard import (
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    gsp_place,
    normalise,
)


# ============================================================================
# GRID DIMENSIONS
# ============================================================================

GRID_ROWS = 64

GRID_COLS = 46


# ============================================================================
# HELPERS
# ============================================================================

def _utc_now() -> str:
    """
    Return current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def _content_hash(
    text: str,
) -> str:
    """
    Return deterministic SHA-256 content identity.
    """

    value = (
        str(text)
        .strip()
        .replace(
            "\r\n",
            "\n",
        )
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _normalise_storage_row(
    row: int,
    rows: int,
) -> int:
    """
    Normalize into 1-based storage rows.

        1 ... rows
    """

    return (
        (
            int(row)
            - 1
        )
        % int(rows)
    ) + 1


def _normalise_storage_col(
    col: int,
    cols: int,
) -> int:
    """
    Normalize into 0-based multilingual columns.

        0 ... cols - 1
    """

    return (
        int(col)
        % int(cols)
    )


# ============================================================================
# MEMORY GRID
# ============================================================================

class MemoryGrid:
    """
    Canonical multilingual spatial memory for CoMpaNeoN.

    MemoryGrid receives text, resolves its language, sends the text
    through tokenizer.py, receives canonical placement information,
    and stores the resulting document and token representation.

    MemoryGrid does not perform crawler traversal.

    Crawlers root through this memory.
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
    ) -> None:

        self.rows = int(
            rows
        )

        self.cols = int(
            cols
        )

        if self.rows <= 0:

            raise ValueError(
                "MemoryGrid rows must be positive."
            )

        if self.cols <= 0:

            raise ValueError(
                "MemoryGrid columns must be positive."
            )

        # --------------------------------------------------------------------
        # DOCUMENT STORAGE
        # --------------------------------------------------------------------

        self.documents: Dict[
            int,
            Dict[str, Any]
        ] = {}

        # --------------------------------------------------------------------
        # CONTENT HASH → DOCUMENT ID
        # --------------------------------------------------------------------

        self.document_hashes: Dict[
            str,
            int
        ] = {}

        # --------------------------------------------------------------------
        # DOCUMENT ID → TOKEN RECORDS
        # --------------------------------------------------------------------

        self.tokens: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------------
        # STORAGE ROW INDEX
        #
        # Used by crawler rooters.
        #
        # Row:
        #
        #     1 ... 64
        # --------------------------------------------------------------------

        self.storage_index: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------------
        # WORD / LANGUAGE GRID INDEX
        #
        # Coordinate:
        #
        #     (row, col)
        #
        # Row:
        #
        #     1 ... 64
        #
        # Column:
        #
        #     0 ... 45
        # --------------------------------------------------------------------

        self.word_index: Dict[
            Tuple[int, int],
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------------
        # LANGUAGE / PHASE INDEX
        #
        # Preserves multilingual tokenizer phase access.
        # --------------------------------------------------------------------

        self.phase_index: Dict[
            int,
            List[Dict[str, Any]]
        ] = {}

        # --------------------------------------------------------------------
        # DOCUMENT COUNTER
        # --------------------------------------------------------------------

        self._next_document_id = 1

        # --------------------------------------------------------------------
        # CRAWLER INSTANCES
        #
        # Lazy initialization prevents circular imports while allowing
        # MemoryGrid to become the shared crawler integration point.
        # --------------------------------------------------------------------

        self._grid_crawler = None

        self._crawler_scheduler = None

        self._crawler_retrieval = None

        self._web_crawler = None

        # --------------------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------------------

        self.documents_added = 0

        self.tokens_added = 0

        self.duplicates_detected = 0

        self.crawler_requests = 0

        self.retrieval_requests = 0

    # ========================================================================
    # GRID DIMENSIONS
    # ========================================================================

    def grid_dimensions(
        self,
    ) -> Tuple[int, int]:
        """
        Return:

            (rows, cols)
        """

        return (
            self.rows,
            self.cols,
        )

    def dimensions(
        self,
    ) -> Dict[str, int]:
        """
        Return canonical grid dimensions.
        """

        return {
            "rows": self.rows,
            "cols": self.cols,
        }

    # ========================================================================
    # LANGUAGE RESOLUTION
    # ========================================================================

    def resolve_language(
        self,
        text: str,
        lang: Optional[str] = None,
    ) -> str:
        """
        Resolve language before text enters the tokenizer and grid.

        Priority:

            1. Explicit language
            2. langdetect
            3. English fallback

        The resulting language is normalized through tokenizer.py.
        """

        if lang:

            return normalize_lang(
                lang
            )

        text = str(
            text
        ).strip()

        if (
            detect_language_code is not None
            and text
        ):

            try:

                detected = (
                    detect_language_code(
                        text
                    )
                )

                return normalize_lang(
                    detected
                )

            except Exception:

                pass

        return normalize_lang(
            "en"
        )

    # ========================================================================
    # TOKENIZER PIPELINE
    # ========================================================================

    def tokenize_text(
        self,
        text: str,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Canonical multilingual tokenizer pipeline.

        MemoryGrid does not recreate language mathematics.

        tokenizer.py remains the linguistic authority.
        """

        resolved_lang = (
            self.resolve_language(
                text,
                lang,
            )
        )

        tokens = tokenize(
            text,
            resolved_lang,
        )

        return {
            "text": text,
            "language": resolved_lang,
            "tokens": tokens,
            "token_count": len(
                tokens
            ),
        }

    # ========================================================================
    # CANONICAL TEXT PLACEMENT
    # ========================================================================

    def placement_for_text(
        self,
        text: str,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determine canonical MemoryGrid entry for text.

        This method does not crawl.

        It determines the canonical entry point into:

            46 × 64 MemoryGrid

        GridCrawler subsequently enters through this placement and applies:

            K = 250
            forward_d = 5
            backward_k = 5
            backward_d = 1

        Those traversal rules remain exclusively inside GridCrawler.
        """

        resolved_lang = (
            self.resolve_language(
                text,
                lang,
            )
        )

        normalized = normalise(
            text,
            resolved_lang,
        )

        Lsum = calculate_lsum(
            normalized,
            resolved_lang,
        )

        Ssum = calculate_ssum(
            normalized,
            resolved_lang,
        )

        c = first_letter_index(
            normalized,
            resolved_lang,
        )

        # --------------------------------------------------------------------
        # CANONICAL PLACEMENT
        #
        # K=0 and D=0 here mean:
        #
        # placement only
        #
        # They are not crawler traversal values.
        #
        # GridCrawler owns actual K/D traversal.
        # --------------------------------------------------------------------

        placement = gsp_place(
            Lsum=Lsum,
            Ssum=Ssum,
            c=c,
            K=0,
            D=0,
            C=self.cols,
            R=self.rows,
        )

        start_row = (
            _normalise_storage_row(
                int(
                    placement[
                        "start_row"
                    ]
                ),
                self.rows,
            )
        )

        # --------------------------------------------------------------------
        # 46-COLUMN RULE
        #
        # The tokenizer / linguistic authority can produce phases beyond
        # the original alphabet width of 26.
        #
        # Therefore MemoryGrid normalizes against self.cols = 46.
        # --------------------------------------------------------------------

        start_col = (
            _normalise_storage_col(
                c,
                self.cols,
            )
        )

        return {
            "normalized": normalized,
            "language": resolved_lang,
            "Lsum": Lsum,
            "Ssum": Ssum,
            "c": c,
            "start_row": start_row,
            "start_col": start_col,
            "placement": placement,
        }

    # ========================================================================
    # DOCUMENT ID
    # ========================================================================

    def _allocate_document_id(
        self,
    ) -> int:

        document_id = (
            self._next_document_id
        )

        self._next_document_id += 1

        return document_id

    # ========================================================================
    # ADD DOCUMENT
    # ========================================================================

    def add_document(
        self,
        text: str,
        lang: Optional[str] = None,
        source: str = "memory",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> int:
        """
        Add a document into canonical CoMpaNeoN memory.

        Pipeline:

            text
              ↓
            language detection
              ↓
            tokenizer.py
              ↓
            placement / keyboard
              ↓
            MemoryGrid 46 × 64
              ↓
            document
              ↓
            storage row
              ↓
            tokens
              ↓
            word / language cells
        """

        text = str(
            text
        ).strip()

        if not text:

            raise ValueError(
                "Cannot index empty text."
            )

        checksum = (
            _content_hash(
                text
            )
        )

        # --------------------------------------------------------------------
        # DUPLICATE DETECTION
        # --------------------------------------------------------------------

        existing_id = (
            self.document_hashes.get(
                checksum
            )
        )

        if existing_id is not None:

            self.duplicates_detected += 1

            return existing_id

        # --------------------------------------------------------------------
        # TOKENIZER
        # --------------------------------------------------------------------

        token_data = (
            self.tokenize_text(
                text,
                lang,
            )
        )

        resolved_lang = (
            token_data[
                "language"
            ]
        )

        # --------------------------------------------------------------------
        # CANONICAL GRID ENTRY
        # --------------------------------------------------------------------

        placement = (
            self.placement_for_text(
                text,
                resolved_lang,
            )
        )

        # --------------------------------------------------------------------
        # DOCUMENT ID
        # --------------------------------------------------------------------

        document_id = (
            self._allocate_document_id()
        )

        # --------------------------------------------------------------------
        # DOCUMENT RECORD
        # --------------------------------------------------------------------

        document = {

            "doc_id":
                document_id,

            "text":
                text,

            "original":
                text,

            "language":
                resolved_lang,

            "lang":
                resolved_lang,

            "source":
                source,

            "content_hash":
                checksum,

            "created_at":
                _utc_now(),

            "placement":
                placement,

            "metadata":
                dict(
                    metadata
                    or {}
                ),

        }

        self.documents[
            document_id
        ] = document

        self.document_hashes[
            checksum
        ] = document_id

        self.documents_added += 1

        # --------------------------------------------------------------------
        # DOCUMENT STORAGE ROW
        #
        # GridCrawler can root from this canonical row.
        # --------------------------------------------------------------------

        storage_row = (
            _normalise_storage_row(
                placement[
                    "start_row"
                ],
                self.rows,
            )
        )

        self.storage_index.setdefault(
            storage_row,
            [],
        ).append(
            document
        )

        # --------------------------------------------------------------------
        # TOKEN STORAGE
        # --------------------------------------------------------------------

        stored_tokens: List[
            Dict[str, Any]
        ] = []

        for position, token in enumerate(
            token_data[
                "tokens"
            ]
        ):

            token_record = (
                self._build_token_record(
                    token=token,
                    position=position,
                    document=document,
                    placement=placement,
                )
            )

            stored_tokens.append(
                token_record
            )

            self._index_token(
                token_record
            )

            self.tokens_added += 1

        self.tokens[
            document_id
        ] = stored_tokens

        return document_id
