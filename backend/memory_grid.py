"""
CoMpaNeoN Memory Grid

Canonical multilingual memory storage layer.

ARCHITECTURE

External / User / AI Text
│
▼
Language Detection
│
▼
tokenizer.py
│
├── normalization
├── language mapping
├── linguistic phase
└── tokenization
│
▼
placement / keyboard
│
▼
MemoryGrid
46 × 64
│
├── documents
├── tokens
├── storage cells
└── word cells
│
▼
Crawler rooters
│
├── GridCrawler
├── WebCrawler
├── other crawlers
└── CrawlerRetrieval

RESPONSIBILITIES

MemoryGrid owns:

- canonical multilingual grid dimensions
- document storage
- token storage
- language preservation
- document metadata
- document identity
- token identity
- calling tokenizer.py
- receiving placement information
- storage lookup

MemoryGrid does NOT own:

- crawler K mathematics
- crawler D mathematics
- backward perturbations
- Elastic Cloud traversal
- external HTTP acquisition
- ranking
- WordChain
- WordUnderstanding
- AI generation

IMPORTANT

GridCrawler is a rooter.

WebCrawler is an acquisition crawler.

MemoryGrid is the shared spatial memory they enter and operate upon.

The canonical grid is:

46 columns
64 rows

The 46-column width exists because the multilingual tokenizer
can produce language phases beyond the original 26-column alphabet
space.

Crawler traversal must therefore use MemoryGrid.cols rather than
assuming 26 columns.
"""

from future import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

============================================================================

OPTIONAL LANGUAGE DETECTION

============================================================================

try:

from langdetect import detect as detect_language_code

except ImportError:

detect_language_code = None

============================================================================

TOKENIZER

============================================================================

from tokenizer import (
normalize_lang,
tokenize,
)

============================================================================

KEYBOARD / PLACEMENT

============================================================================

from keyboard import (
calculate_lsum,
calculate_ssum,
first_letter_index,
gsp_place,
normalise,
)

============================================================================

GRID DIMENSIONS

============================================================================

GRID_ROWS = 64

GRID_COLS = 46

============================================================================

HELPERS

============================================================================

def _utc_now() -> str:
"""
Return UTC timestamp.
"""

return datetime.now(
    timezone.utc
).isoformat()

def _content_hash(
text: str,
) -> str:
"""
Deterministic SHA-256 content identity.
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
Normalize row into 1-based storage coordinates.

Valid rows:

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
Normalize column into 0-based coordinates.

Valid columns:

    0 ... cols - 1
"""

return (
    int(col)
    % int(cols)
)

============================================================================

MEMORY GRID

============================================================================

class MemoryGrid:
"""
Canonical multilingual CoMpaNeoN memory space.

The MemoryGrid stores documents and their tokenizer-derived
linguistic representation.

It does not perform crawler traversal.

Crawlers use this class as their shared spatial memory.
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
    # CONTENT HASH INDEX
    # --------------------------------------------------------------------

    self.document_hashes: Dict[
        str,
        int
    ] = {}

    # --------------------------------------------------------------------
    # TOKEN STORAGE
    # --------------------------------------------------------------------

    self.tokens: Dict[
        int,
        List[Dict[str, Any]]
    ] = {}

    # --------------------------------------------------------------------
    # STORAGE ROW INDEX
    #
    # GridCrawler.get_tokens_at_storage(row)
    #
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
    # --------------------------------------------------------------------

    self.word_index: Dict[
        Tuple[int, int],
        List[Dict[str, Any]]
    ] = {}

    # --------------------------------------------------------------------
    # DOCUMENT ID COUNTER
    # --------------------------------------------------------------------

    self._next_document_id = 1

    # --------------------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------------------

    self.documents_added = 0

    self.tokens_added = 0

    self.duplicates_detected = 0

# ========================================================================
# LANGUAGE
# ========================================================================

def resolve_language(
    self,
    text: str,
    lang: Optional[str] = None,
) -> str:
    """
    Resolve canonical language.

    Priority:

        1. Explicit supplied language.
        2. langdetect.
        3. English fallback.

    The final language identifier is normalized through tokenizer.py.
    """

    if lang:

        return normalize_lang(
            lang
        )

    if (
        detect_language_code is not None
        and str(text).strip()
    ):

        try:

            detected = (
                detect_language_code(
                    str(text)
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
# GRID DIMENSIONS
# ========================================================================

def dimensions(
    self,
) -> Dict[str, int]:
    """
    Return canonical MemoryGrid dimensions.
    """

    return {
        "rows": self.rows,
        "cols": self.cols,
    }

# ========================================================================
# CANONICAL TOKENIZER PIPELINE
# ========================================================================

def tokenize_text(
    self,
    text: str,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send text through the canonical tokenizer pipeline.

    MemoryGrid does not independently create linguistic mathematics.

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
# TEXT PLACEMENT
# ========================================================================

def placement_for_text(
    self,
    text: str,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve the canonical grid entry point for text.

    IMPORTANT:

    This method does NOT crawl.

    It only determines where the text belongs within the
    46 × 64 MemoryGrid.

    GridCrawler later uses this entry point as the beginning of
    its own K/D traversal.
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
    # K and D here are neutral placement values.
    #
    # They do NOT represent crawler K/D traversal.
    #
    # Crawler K/D remains inside GridCrawler.
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

    start_row = int(
        placement[
            "start_row"
        ]
    )

    start_col = (
        int(c)
        % self.cols
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
    Add a document to MemoryGrid.

    Pipeline:

        text
          ↓
        language resolution
          ↓
        tokenizer.py
          ↓
        canonical placement
          ↓
        document storage
          ↓
        storage row index
          ↓
        word/language coordinate index

    Returns the canonical document ID.
    """

    text = str(
        text
    ).strip()

    if not text:

        raise ValueError(
            "Cannot index empty text."
        )

    checksum = _content_hash(
        text
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
    # CANONICAL TEXT ENTRY
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
            metadata
            or {},

    }

    self.documents[
        document_id
    ] = document

    self.document_hashes[
        checksum
    ] = document_id

    self.documents_added += 1

    # --------------------------------------------------------------------
    # STORAGE ROW
    #
    # The document itself is visible from its canonical GSP row.
    #
    # GridCrawler uses this as one of its rooting access paths.
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
        []
    ).append(
        document
    )

    # --------------------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------------------

    stored_tokens = []

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
    Convert tokenizer output into a MemoryGrid token record.

    The tokenizer's own coordinate information is preserved when
    available.

    MemoryGrid does not invent an alternative language mapping.
    """

    if isinstance(
        token,
        dict,
    ):

        token_value = (
            token.get(
                "token"
            )
            or token.get(
                "text"
            )
            or token.get(
                "original"
            )
            or ""
        )

        token_data = dict(
            token
        )

    else:

        token_value = str(
            token
        )

        token_data = {}

    # --------------------------------------------------------------------
    # TOKENIZER-PROVIDED COORDINATES
    #
    # If tokenizer later exposes canonical row/column fields,
    # MemoryGrid consumes them.
    #
    # --------------------------------------------------------------------

    tokenizer_row = (
        token_data.get(
            "row"
        )
        or token_data.get(
            "storage_row"
        )
    )

    tokenizer_col = (
        token_data.get(
            "col"
        )
        or token_data.get(
            "column"
        )
        or token_data.get(
            "phase"
        )
        or token_data.get(
            "language_phase"
        )
    )

    # --------------------------------------------------------------------
    # FALLBACK
    #
    # Document canonical placement is used only if tokenizer has
    # not yet supplied a token coordinate.
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
    Index a token into MemoryGrid access structures.
    """

    row = int(
        token[
            "row"
        ]
    )

    col = int(
        token[
            "col"
        ]
    )

    # --------------------------------------------------------------------
    # STORAGE ROW INDEX
    # --------------------------------------------------------------------

    self.storage_index.setdefault(
        row,
        []
    ).append(
        token
    )

    # --------------------------------------------------------------------
    # WORD / LANGUAGE GRID INDEX
    # --------------------------------------------------------------------

    self.word_index.setdefault(
        (
            row,
            col,
        ),
        []
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
    Return all documents/tokens accessible from a storage row.

    Used by GridCrawler as one of its rooting access mechanisms.
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
    Return tokens at a multilingual word-grid coordinate.

    Coordinates are normalized into:

        rows: 1 ... 64
        cols: 0 ... 45
    """

    row = (
        _normalise_storage_row(
    