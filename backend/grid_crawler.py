"""
CoMpaNeoN Grid Crawler
======================

Query-aware deterministic grid traversal layer.

Architecture
------------

User / AI Query
    ↓
placement.py
    ├── identity/message placement
    ├── room placement
    ├── workspace placement
    └── full-text placement
    ↓
tokenizer.py
    ├── language detection
    ├── tokenization
    ├── linguistic normalization
    ├── alphabet mapping
    └── word coordinates
    ↓
grid_crawler.py
    ├── query entry points
    ├── token entry points
    ├── full-query entry point
    ├── K traversal
    ├── forward movement
    ├── backward perturbation
    └── Elastic Cloud
    ↓
MemoryGrid
    ├── Letter Grid
    ├── Word Grid
    └── Full-text Storage Grid
    ↓
Candidate Board
    ↓
crawler_retrieval.py
    ↓
ranking.py
    ↓
PromptManager
    ↓
AI Model / AI Models
    ↓
Response

Responsibilities
----------------

GridCrawler:

    - receives a user's query
    - tokenizes the query
    - creates deterministic grid entry points
    - uses placement.py where placement context is available
    - uses tokenizer linguistic coordinates
    - enters MemoryGrid through canonical GSP positions
    - traverses K positions
    - applies forward movement
    - applies backward perturbations
    - applies Elastic Cloud
    - collects candidates
    - returns query-aware candidate metadata

GridCrawler does NOT:

    - rank final results
    - decide final intent
    - generate prompts
    - generate answers
    - replace intent_analyzer.py
    - replace placement.py
    - replace tokenizer.py
    - perform AI reasoning

IMPORTANT
---------

K and D belong to crawling.

They do NOT belong to:

    - tokenizer.py
    - letter placement
    - word placement
    - MemoryGrid document placement
    - prompt generation
    - AI model inference

The crawler is a retrieval traversal layer.

It prepares candidate knowledge for the AI.

Canonical crawler configuration:

    K             = 250
    forward_d     = 5
    backward_k    = 5
    backward_d    = 1
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


# ============================================================================
# KEYBOARD / GSP
# ============================================================================

from keyboard import (
    gsp_place,
    elastic_cloud,
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    normalise,
)


# ============================================================================
# TOKENIZER
# ============================================================================

from tokenizer import (
    tokenize,
    normalize_lang,
)


# ============================================================================
# PLACEMENT
# ============================================================================

try:

    from placement import (
        Placement,
    )

except ImportError:

    Placement = None


# ============================================================================
# CRAWLER CONFIGURATION
# ============================================================================

CRAWLER_K = 250

FORWARD_D = 5

BACKWARD_K = 5

BACKWARD_D = 1

DEFAULT_LIMIT = 250

ELASTIC_RADIUS = 1

ELASTIC_FIRST_LETTER_RADIUS = 1


# ============================================================================
# HELPERS
# ============================================================================

def _normalise_cell(
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> Tuple[int, int]:

    return (
        ((int(row) - 1) % rows) + 1,
        int(col) % cols,
    )


def _candidate_key(
    candidate: Dict[str, Any],
) -> Tuple[Any, Any]:

    return (
        candidate.get(
            "doc_id"
        ),

        candidate.get(
            "original"
        ),
    )


def _document_key(
    candidate: Dict[str, Any],
) -> Any:

    return candidate.get(
        "doc_id"
    )


# ============================================================================
# GRID CRAWLER
# ============================================================================

class GridCrawler:
    """
    Query-aware deterministic crawler.

    The crawler converts a query into multiple independent
    retrieval routes.

    Query routes:

        ROUTE 1
            Full-query GSP entry.

        ROUTE 2
            Individual token GSP entries.

        ROUTE 3
            Tokenizer Word Grid coordinates.

        ROUTE 4
            MemoryGrid Letter Grid coordinates.

        ROUTE 5
            Elastic Cloud around canonical token positions.

    These routes remain independent.

    Candidate merging happens here.

    Final ranking belongs above GridCrawler.
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        memory_grid: Any,
        placement: Optional[Any] = None,
        K: int = CRAWLER_K,
        forward_d: int = FORWARD_D,
        backward_k: int = BACKWARD_K,
        backward_d: int = BACKWARD_D,
        elastic_radius: int = ELASTIC_RADIUS,
        elastic_first_letter_radius: int = (
            ELASTIC_FIRST_LETTER_RADIUS
        ),
    ) -> None:

        self.memory = memory_grid

        self.placement = placement

        self.K = int(
            K
        )

        self.forward_d = int(
            forward_d
        )

        self.backward_k = int(
            backward_k
        )

        self.backward_d = int(
            backward_d
        )

        self.elastic_radius = int(
            elastic_radius
        )

        self.elastic_first_letter_radius = int(
            elastic_first_letter_radius
        )

    # ========================================================================
    # GRID DIMENSIONS
    # ========================================================================

    @property
    def rows(
        self,
    ) -> int:

        return int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

    # ------------------------------------------------------------------------

    @property
    def cols(
        self,
    ) -> int:

        return int(
            getattr(
                self.memory,
                "cols",
                26,
            )
        )

    # ========================================================================
    # QUERY LANGUAGE
    # ========================================================================

    def resolve_language(
        self,
        lang: str = "en",
    ) -> str:
        """
        Normalize the query language.

        Language ownership remains with tokenizer.py.
        """

        return normalize_lang(
            lang
        )

    # ========================================================================
    # QUERY TOKENIZATION
    # ========================================================================

    def tokenize_query(
        self,
        query: str,
        lang: str = "en",
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Tokenize a user query.

        Tokenizer remains the owner of:

            - token boundaries
            - language normalization
            - stems
            - alphabet indices
            - word-grid coordinates
            - local dialect token representation
        """

        if not query:

            return []

        active_language = self.resolve_language(
            lang
        )

        return tokenize(
            query,
            active_language,
        )

    # ========================================================================
    # CANONICAL GSP ENTRY
    # ========================================================================

    def canonical_start_row(
        self,
        text: str,
        lang: str = "en",
        uid: Optional[Any] = None,
        placement_type: str = "query",
    ) -> int:
        """
        Resolve canonical GSP start row.

        If placement.py is available and can handle the supplied
        placement type, it may be used as the higher-level placement
        authority.

        Otherwise keyboard.py remains the canonical GSP fallback.

        This preserves the architecture:

            placement.py
                owns contextual placement.

            keyboard.py
                owns GSP mathematics.
        """

        if not text:

            return 0

        active_language = self.resolve_language(
            lang
        )

        # --------------------------------------------------------------------
        # PLACEMENT LAYER
        # --------------------------------------------------------------------

        if self.placement is not None:

            if hasattr(
                self.placement,
                "place",
            ):

                try:

                    result = self.placement.place(
                        text=text,
                        lang=active_language,
                        uid=uid,
                        placement_type=placement_type,
                    )

                    if isinstance(
                        result,
                        dict,
                    ):

                        start_row = result.get(
                            "start_row"
                        )

                        if start_row:

                            return int(
                                start_row
                            )

                except (
                    TypeError,
                    AttributeError,
                    ValueError,
                ):

                    pass

        # --------------------------------------------------------------------
        # KEYBOARD FALLBACK
        # --------------------------------------------------------------------

        normalized = normalise(
            text,
            active_language,
        )

        Lsum = calculate_lsum(
            normalized,
            active_language,
        )

        Ssum = calculate_ssum(
            normalized,
            active_language,
        )

        c = first_letter_index(
            normalized,
            active_language,
        )

        result = gsp_place(
            Lsum=Lsum,
            Ssum=Ssum,
            c=c,
            K=0,
            D=self.forward_d,
            C=self.cols,
            R=self.rows,
        )

        return int(
            result.get(
                "start_row",
                0,
            )
        )

    # ========================================================================
    # QUERY GSP INFORMATION
    # ========================================================================

    def query_gsp(
        self,
        query: str,
        lang: str = "en",
        uid: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build canonical GSP metadata for a complete query.

        The query is treated as a complete-text retrieval entry.

        This is independent from per-token crawling.
        """

        active_language = self.resolve_language(
            lang
        )

        normalized = normalise(
            query,
            active_language,
        )

        Lsum = calculate_lsum(
            normalized,
            active_language,
        )

        Ssum = calculate_ssum(
            normalized,
            active_language,
        )

        c = first_letter_index(
            normalized,
            active_language,
        )

        start_row = self.canonical_start_row(
            text=query,
            lang=active_language,
            uid=uid,
            placement_type="query",
        )

        return {

            "query": query,

            "normalized": normalized,

            "lang": active_language,

            "L": int(
                Lsum
            ),

            "S": int(
                Ssum
            ),

            "c": int(
                c
            ),

            "start_row": int(
                start_row
            ),

            "rows": self.rows,

            "cols": self.cols,
        }

    # ========================================================================
    # FORWARD POSITION
    # ========================================================================

    def forward_row(
        self,
        start_row: int,
        k: int,
    ) -> int:

        return (
            (
                int(start_row)
                - 1
                + int(k)
                * self.forward_d
            )
            % self.rows
        ) + 1

    # ========================================================================
    # BACKWARD PERTURBATIONS
    # ========================================================================

    def backward_rows(
        self,
        forward_row: int,
    ) -> List[
        int
    ]:

        rows: List[
            int
        ] = []

        for perturbation in range(
            1,
            self.backward_k + 1,
        ):

            row = (
                (
                    int(forward_row)
                    - 1
                    - perturbation
                    * self.backward_d
                )
                % self.rows
            ) + 1

            rows.append(
                row
            )

        return rows

    # ========================================================================
    # TRAVERSAL PATH
    # ========================================================================

    def traversal_path(
        self,
        start_row: int,
        start_col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Build the deterministic traversal path.

        Every K split contains:

            forward position

            then:

                backward -1
                backward -2
                backward -3
                backward -4
                backward -5
        """

        path: List[
            Dict[str, Any]
        ] = []

        seen: Set[
            Tuple[int, int]
        ] = set()

        for k in range(
            self.K
        ):

            forward = self.forward_row(
                start_row,
                k,
            )

            col = (
                int(start_col)
                + k
            ) % self.cols

            forward_cell = (
                forward,
                col,
            )

            if forward_cell not in seen:

                seen.add(
                    forward_cell
                )

                path.append({

                    "row": forward,

                    "col": col,

                    "k": k,

                    "direction": "forward",

                    "perturbation_k": 0,
                })

            # --------------------------------------------------------------
            # BACKWARD PERTURBATION
            # --------------------------------------------------------------

            for perturbation_k, backward in enumerate(
                self.backward_rows(
                    forward
                ),
                start=1,
            ):

                cell = (
                    backward,
                    col,
                )

                if cell in seen:

                    continue

                seen.add(
                    cell
                )

                path.append({

                    "row": backward,

                    "col": col,

                    "k": k,

                    "direction": "backward",

                    "perturbation_k": perturbation_k,
                })

        return path

    # ========================================================================
    # ELASTIC CLOUD
    # ========================================================================

    def elastic_candidates(
        self,
        L: int,
        S: int,
        c: int,
    ) -> List[
        Dict[str, int]
    ]:

        return elastic_cloud(
            L=L,
            S=S,
            c=c,
            radius=self.elastic_radius,
            first_letter_radius=(
                self.elastic_first_letter_radius
            ),
            C=self.cols,
            R=self.rows,
        )

    # ========================================================================
    # MEMORY CELL ACCESS
    # ========================================================================

    def _read_cell(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Read every available MemoryGrid route intersecting
        the crawler position.
        """

        results: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------------------
        # FULL-TEXT STORAGE GRID
        # --------------------------------------------------------------------

        if hasattr(
            self.memory,
            "get_tokens_at_storage",
        ):

            results.extend(
                self.memory.get_tokens_at_storage(
                    row
                )
            )

        # --------------------------------------------------------------------
        # WORD GRID
        # --------------------------------------------------------------------

        if hasattr(
            self.memory,
            "get_tokens_at_word",
        ):

            results.extend(
                self.memory.get_tokens_at_word(
                    row,
                    col,
                )
            )

        # --------------------------------------------------------------------
        # LEGACY
        # --------------------------------------------------------------------

        elif hasattr(
            self.memory,
            "get_tokens_at",
        ):

            results.extend(
                self.memory.get_tokens_at(
                    row,
                    col,
                )
            )

        return results

    # ========================================================================
    # LETTER ROUTE
    # ========================================================================

    def crawl_token_letters(
        self,
        token_info: Dict[str, Any],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Retrieve candidates directly through the Letter Grid.

        This is important because the AI must not depend only
        on whole-text or whole-word matching.

        Tokenized linguistic information is also a retrieval path.
        """

        results: List[
            Dict[str, Any]
        ] = []

        seen: Set[
            Tuple[Any, Any]
        ] = set()

        if not hasattr(
            self.memory,
            "get_tokens_at_letter",
        ):

            return results

        letters = token_info.get(
            "letter",
            [],
        )

        for letter_index in letters:

            entries = (
                self.memory.get_tokens_at_letter(
                    int(letter_index)
                )
            )

            for entry in entries:

                key = _candidate_key(
                    entry
                )

                if key in seen:

                    continue

                seen.add(
                    key
                )

                result = dict(
                    entry
                )

                result["crawl"] = {

                    "route": "letter",

                    "letter_index": int(
                        letter_index
                    ),
                }

                results.append(
                    result
                )

        return results

    # ========================================================================
    # WORD ROUT