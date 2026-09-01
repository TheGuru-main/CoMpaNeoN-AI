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

    # ========================================================================
    # TOKEN RECORD
    # ========================================================================

    def _build_token_record(
        self,
        token: Any,
        position: int,
        document: Dict[str, Any],
        placement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preserve tokenizer information inside MemoryGrid.

        The tokenizer remains authoritative for token coordinates,
        language phases and multilingual representation.
        """

        if isinstance(
            token,
            dict,
        ):

            token_data = dict(
                token
            )

            token_value = (
                token_data.get(
                    "token"
                )
                or token_data.get(
                    "text"
                )
                or token_data.get(
                    "original"
                )
                or ""
            )

        else:

            token_data = {}

            token_value = str(
                token
            )

        # --------------------------------------------------------------------
        # TOKENIZER ROW
        # --------------------------------------------------------------------

        tokenizer_row = (
            token_data.get(
                "row"
            )
        )

        if tokenizer_row is None:

            tokenizer_row = (
                token_data.get(
                    "storage_row"
                )
            )

        # --------------------------------------------------------------------
        # TOKENIZER COLUMN / PHASE
        #
        # This is where multilingual phases can extend beyond 26.
        #
        # MemoryGrid accepts tokenizer information and normalizes it into
        # the canonical 46-column grid.
        # --------------------------------------------------------------------

        tokenizer_col = (
            token_data.get(
                "col"
            )
        )

        if tokenizer_col is None:

            tokenizer_col = (
                token_data.get(
                    "column"
                )
            )

        if tokenizer_col is None:

            tokenizer_col = (
                token_data.get(
                    "phase"
                )
            )

        if tokenizer_col is None:

            tokenizer_col = (
                token_data.get(
                    "language_phase"
                )
            )

        # --------------------------------------------------------------------
        # FALLBACK TO DOCUMENT ENTRY
        # --------------------------------------------------------------------

        row = (
            tokenizer_row
            if tokenizer_row is not None
            else placement[
                "start_row"
            ]
        )

        col = (
            tokenizer_col
            if tokenizer_col is not None
            else placement[
                "start_col"
            ]
        )

        row = (
            _normalise_storage_row(
                row,
                self.rows,
            )
        )

        col = (
            _normalise_storage_col(
                col,
                self.cols,
            )
        )

        language_phase = (
            token_data.get(
                "language_phase"
            )
        )

        if language_phase is None:

            language_phase = (
                token_data.get(
                    "phase"
                )
            )

        if language_phase is None:

            language_phase = col

        return {

            "doc_id":
                document[
                    "doc_id"
                ],

            "token":
                token_value,

            "original":
                token_data.get(
                    "original",
                    token_value,
                ),

            "position":
                position,

            "row":
                row,

            "col":
                col,

            "language_phase":
                language_phase,

            "language":
                document[
                    "language"
                ],

            "lang":
                document[
                    "lang"
                ],

            "source":
                document[
                    "source"
                ],

            "tokenizer":
                token_data,

        }

    # ========================================================================
    # TOKEN INDEXING
    # ========================================================================

    def _index_token(
        self,
        token: Dict[str, Any],
    ) -> None:
        """
        Index token into all MemoryGrid access structures.
        """

        row = (
            _normalise_storage_row(
                token[
                    "row"
                ],
                self.rows,
            )
        )

        col = (
            _normalise_storage_col(
                token[
                    "col"
                ],
                self.cols,
            )
        )

        phase = (
            _normalise_storage_col(
                token.get(
                    "language_phase",
                    col,
                ),
                self.cols,
            )
        )

        # --------------------------------------------------------------------
        # STORAGE ROW
        # --------------------------------------------------------------------

        self.storage_index.setdefault(
            row,
            [],
        ).append(
            token
        )

        # --------------------------------------------------------------------
        # WORD / LANGUAGE CELL
        # --------------------------------------------------------------------

        self.word_index.setdefault(
            (
                row,
                col,
            ),
            [],
        ).append(
            token
        )

        # --------------------------------------------------------------------
        # LANGUAGE PHASE
        # --------------------------------------------------------------------

        self.phase_index.setdefault(
            phase,
            [],
        ).append(
            token
        )

    # ========================================================================
    # STORAGE LOOKUP
    # ========================================================================

    def get_tokens_at_storage(
        self,
        row: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return all entries accessible from a storage row.

        Used by GridCrawler rooting.
        """

        row = (
            _normalise_storage_row(
                row,
                self.rows,
            )
        )

        return list(
            self.storage_index.get(
                row,
                [],
            )
        )

    # ========================================================================
    # WORD LOOKUP
    # ========================================================================

    def get_tokens_at_word(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return tokens from a multilingual word coordinate.

        Rows:

            1 ... 64

        Columns:

            0 ... 45
        """

        row = (
            _normalise_storage_row(
                row,
                self.rows,
            )
        )

        col = (
            _normalise_storage_col(
                col,
                self.cols,
            )
        )

        return list(
            self.word_index.get(
                (
                    row,
                    col,
                ),
                [],
            )
        )

    # ========================================================================
    # LANGUAGE PHASE LOOKUP
    # ========================================================================

    def get_tokens_at_phase(
        self,
        phase: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return tokens associated with a tokenizer linguistic phase.

        The phase is normalized against the canonical 46-column width.
        """

        phase = (
            _normalise_storage_col(
                phase,
                self.cols,
            )
        )

        return list(
            self.phase_index.get(
                phase,
                [],
            )
        )

    # ========================================================================
    # LEGACY LOOKUP
    # ========================================================================

    def get_tokens_at(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Compatibility lookup for existing callers.
        """

        return self.get_tokens_at_word(
            row,
            col,
        )

    # ========================================================================
    # DOCUMENT LOOKUP
    # ========================================================================

    def get_document(
        self,
        doc_id: int,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Retrieve one document.
        """

        return self.documents.get(
            int(doc_id)
        )

    # ========================================================================
    # DOCUMENT TOKENS
    # ========================================================================

    def get_document_tokens(
        self,
        doc_id: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Retrieve all token records belonging to a document.
        """

        return list(
            self.tokens.get(
                int(doc_id),
                [],
            )
        )

    # ========================================================================
    # GRID CRAWLER
    # ========================================================================

    def get_grid_crawler(
        self,
    ) -> Any:
        """
        Lazily initialize GridCrawler.

        GridCrawler remains the rooter and owns:

            K
            forward_d
            backward_k
            backward_d
            Elastic Cloud traversal
        """

        if self._grid_crawler is None:

            from grid_crawler import (
                GridCrawler
            )

            self._grid_crawler = (
                GridCrawler(
                    self
                )
            )

        return self._grid_crawler

    # ========================================================================
    # CRAWLER SCHEDULER
    # ========================================================================

    def get_crawler_scheduler(
        self,
    ) -> Any:
        """
        Lazily initialize CrawlerScheduler.
        """

        if self._crawler_scheduler is None:

            from crawler_scheduler import (
                CrawlerScheduler
            )

            self._crawler_scheduler = (
                CrawlerScheduler()
            )

        return self._crawler_scheduler

    # ========================================================================
    # CRAWLER RETRIEVAL
    # ========================================================================

    def get_crawler_retrieval(
        self,
    ) -> Any:
        """
        Lazily initialize CrawlerRetrieval.

        The concrete retrieval API can continue evolving independently
        without moving ranking responsibility into MemoryGrid.
        """

        if self._crawler_retrieval is None:

            from crawler_retrieval import (
                CrawlerRetrieval
            )

            self._crawler_retrieval = (
                CrawlerRetrieval(
                    self
                )
            )

        return self._crawler_retrieval

    # ========================================================================
    # WEB CRAWLER
    # ========================================================================

    def get_web_crawler(
        self,
    ) -> Any:
        """
        Lazily initialize WebCrawler.

        WebCrawler acquires external information and dumps documents into
        this MemoryGrid.
        """

        if self._web_crawler is None:

            from web_crawler import (
                WebCrawler
            )

            self._web_crawler = (
                WebCrawler(
                    self
                )
            )

        return self._web_crawler

    # ========================================================================
    # ROOT TEXT THROUGH GRID CRAWLER
    # ========================================================================

    def crawl_text(
        self,
        text: str,
        lang: Optional[str] = None,
        limit: int = 250,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Root a text query through GridCrawler.

        MemoryGrid provides the multilingual 46 × 64 entry.

        GridCrawler then applies its own traversal rules.
        """

        placement = (
            self.placement_for_text(
                text,
                lang,
            )
        )

        crawler = (
            self.get_grid_crawler()
        )

        self.crawler_requests += 1

        return crawler.crawl(
            start_row=placement[
                "start_row"
            ],
            start_col=placement[
                "start_col"
            ],
            L=placement[
                "Lsum"
            ],
            S=placement[
                "Ssum"
            ],
            c=placement[
                "c"
            ],
            limit=limit,
        )

    # ========================================================================
    # RETRIEVAL ENTRY
    # ========================================================================

    def retrieve(
        self,
        query: str,
        lang: Optional[str] = None,
        limit: int = 250,
        **kwargs: Any,
    ) -> Any:
        """
        Enter CoMpaNeoN retrieval through the shared MemoryGrid.

        CrawlerRetrieval remains responsible for retrieval orchestration.
        """

        retrieval = (
            self.get_crawler_retrieval()
        )

        self.retrieval_requests += 1

        return retrieval.retrieve(
            query=query,
            lang=lang,
            limit=limit,
            **kwargs,
        )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def stats(
        self,
    ) -> Dict[str, Any]:
        """
        Return MemoryGrid statistics.
        """

        return {

            "rows":
                self.rows,

            "cols":
                self.cols,

            "documents":
                len(
                    self.documents
                ),

            "tokens":
                self.tokens_added,

            "documents_added":
                self.documents_added,

            "duplicates_detected":
                self.duplicates_detected,

            "storage_rows":
                len(
                    self.storage_index
                ),

            "word_cells":
                len(
                    self.word_index
                ),

            "language_phases":
                len(
                    self.phase_index
                ),

            "crawler_requests":
                self.crawler_requests,

            "retrieval_requests":
                self.retrieval_requests,

        }


# ============================================================================
# DEVELOPMENT TEST
# ============================================================================

if __name__ == "__main__":

    memory = MemoryGrid()

    document_id = (
        memory.add_document(
            text=(
                "CoMpaNeoN is a multilingual "
                "organizational AI architecture."
            ),
            lang="en",
            source="development",
        )
    )

    print(
        "CoMpaNeoN MemoryGrid"
    )

    print(
        memory.dimensions()
    )

    print(
        {
            "document_id":
                document_id
        }
    )

    print(
        memory.stats()
    )