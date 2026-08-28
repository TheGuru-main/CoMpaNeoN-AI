"""
CoMpaNeoN Grid Crawler
======================

Grid traversal layer for CoMpaNeoN.

Responsibilities
----------------
- Enter the grid through a canonical GSP start_row.
- Traverse forward through 250 K positions.
- Apply forward_d = 5.
- Apply backward perturbation of 5 positions at EVERY K split.
- Use backward_d = 1.
- Call Elastic Cloud around crawler jumps.
- Collect candidates from MemoryGrid.
- Index crawled external candidates into MemoryGrid.
- Preserve language information.
- Provide candidates for higher-level retrieval/training.

Architecture
------------
keyboard.py
    Owns canonical GSP calculation:
        start_row = ((L + S - 1) % R) + 1

tokenizer.py
    Owns tokenization and linguistic mapping.

MemoryGrid
    Owns indexed knowledge and retrieval.

grid_crawler.py
    Owns crawler traversal only.

crawler_retrieval.py
    Owns retrieval orchestration/ranking.

IMPORTANT
---------
K and D belong to crawling.

They do NOT belong to:
    - tokenizer.py
    - letter placement
    - word placement
    - ordinary lexical indexing
    - MemoryGrid word-grid placement

Crawler configuration
---------------------
    K             = 250
    forward_d     = 5
    backward_k    = 5
    backward_d    = 1

For every forward K:

    forward position
        ↓
    backward -1
        ↓
    backward -2
        ↓
    backward -3
        ↓
    backward -4
        ↓
    backward -5

Elastic Cloud is evaluated during traversal so that neighbouring
candidate cells can also contribute candidates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from keyboard import (
    gsp_place,
    elastic_cloud,
    calculate_lsum,
    calculate_ssum,
    first_letter_index,
    normalise,
)


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
        candidate.get("doc_id"),
        candidate.get("original"),
    )


# ============================================================================
# GRID CRAWLER
# ============================================================================

class GridCrawler:
    """
    Deterministic crawler over the CoMpaNeoN memory grid.

    The crawler does not calculate a new linguistic representation.

    It receives or derives the canonical GSP start_row through
    keyboard.py and then performs crawler traversal.
    """

    def __init__(
        self,
        memory_grid: Any,
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

        self.K = int(K)

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
    # CANONICAL GSP ENTRY
    # ========================================================================

    def canonical_start_row(
        self,
        word: str,
        lang: str = "en",
    ) -> int:
        """
        Calculate the canonical GSP start row through keyboard.py.

        Formula:

            start_row =
                ((Lsum + Ssum - 1) % R) + 1

        The crawler does not redefine this formula.
        """

        lang = (
            lang or "en"
        ).strip().lower()

        normalized = normalise(
            word,
            lang,
        )

        Lsum = calculate_lsum(
            normalized,
            lang,
        )

        Ssum = calculate_ssum(
            normalized,
            lang,
        )

        R = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        result = gsp_place(
            Lsum=Lsum,
            Ssum=Ssum,
            c=first_letter_index(
                normalized,
                lang,
            ),
            K=0,
            D=self.forward_d,
            C=int(
                getattr(
                    self.memory,
                    "cols",
                    26,
                )
            ),
            R=R,
        )

        return int(
            result["start_row"]
        )

    # ========================================================================
    # FORWARD POSITION
    # ========================================================================

    def forward_row(
        self,
        start_row: int,
        k: int,
    ) -> int:
        """
        Calculate the forward crawler row.

            forward_row =
                ((start_row - 1 + k * forward_d) % R) + 1
        """

        R = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        return (
            (
                int(start_row)
                - 1
                + int(k)
                * self.forward_d
            )
            % R
        ) + 1

    # ========================================================================
    # BACKWARD PERTURBATIONS
    # ========================================================================

    def backward_rows(
        self,
        forward_row: int,
    ) -> List[int]:
        """
        Return the five backward perturbation rows belonging to
        the current forward K split.

        For backward_k = 5:

            forward_row - 1
            forward_row - 2
            forward_row - 3
            forward_row - 4
            forward_row - 5

        The traversal wraps through the 64-row storage dimension.
        """

        R = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        rows = []

        for backward_k in range(
            1,
            self.backward_k + 1,
        ):

            row = (
                (
                    int(forward_row)
                    - 1
                    - backward_k
                    * self.backward_d
                )
                % R
            ) + 1

            rows.append(
                row
            )

        return rows

    # ========================================================================
    # CRAWLER PATH
    # ========================================================================

    def traversal_path(
        self,
        start_row: int,
        start_col: int,
    ) -> List[Dict[str, Any]]:
        """
        Build the deterministic crawler path.

        Each K split produces:

            1 forward cell
            5 backward perturbation cells

        Therefore:

            250 × 6 = 1500

        primary traversal positions before Elastic Cloud candidates.
        """

        rows = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        cols = int(
            getattr(
                self.memory,
                "cols",
                26,
            )
        )

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
            ) % cols

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
            # BACKWARD K = 5 AT THIS SAME FORWARD SPLIT
            # --------------------------------------------------------------

            for perturbation_k, backward in enumerate(
                self.backward_rows(
                    forward
                ),
                start=1,
            ):

                backward_cell = (
                    backward,
                    col,
                )

                if backward_cell in seen:
                    continue

                seen.add(
                    backward_cell
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
        """
        Call keyboard.py Elastic Cloud.

        Elastic Cloud is used during crawling to gather neighbouring
        candidate cells.

        It does not replace the canonical traversal.
        """

        C = int(
            getattr(
                self.memory,
                "cols",
                26,
            )
        )

        R = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        return elastic_cloud(
            L=L,
            S=S,
            c=c,
            radius=self.elastic_radius,
            first_letter_radius=(
                self.elastic_first_letter_radius
            ),
            C=C,
            R=R,
        )

    # ========================================================================
    # MEMORY GRID CELL ACCESS
    # ========================================================================

    def _read_cell(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Read candidates from a MemoryGrid cell.

        MemoryGrid remains the owner of storage.
        """

        results: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # Preferred explicit storage accessor.
        # --------------------------------------------------------------

        if hasattr(
            self.memory,
            "get_tokens_at_storage",
        ):

            results.extend(
                self.memory.get_tokens_at_storage(
                    row
                )
            )

        # --------------------------------------------------------------
        # Word-grid lookup can contribute candidates when the crawler
        # intersects an A×A word coordinate.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Legacy compatibility.
        # --------------------------------------------------------------

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
    # CRAWL CELL
    # ========================================================================

    def crawl_cell(
        self,
        row: int,
        col: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Retrieve candidates from one grid position.
        """

        rows = int(
            getattr(
                self.memory,
                "rows",
                64,
            )
        )

        cols = int(
            getattr(
                self.memory,
                "cols",
                26,
            )
        )

        row, col = _normalise_cell(
            row,
            col,
            rows,
            cols,
        )

        return self._read_cell(
            row,
            col,
        )

    # ========================================================================
    # CRAWL
    # ========================================================================

    def crawl(
        self,
        start_row: int,
        start_col: int,
        L: Optional[int] = None,
        S: Optional[int] = None,
        c: Optional[int] = None,
        limit: int = DEFAULT_LIMIT,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Crawl from a canonical GSP entry point.

        Traversal order:

            forward K
            backward perturbations
            Elastic Cloud

        Candidates are deduplicated by document/token identity.
        """

        candidates: List[
            Dict[str, Any]
        ] = []

        seen_candidates: Set[
            Tuple[Any, Any]
        ] = set()

        path = self.traversal_path(
            start_row,
            start_col,
        )

        for position in path:

            row = position["row"]
            col = position["col"]

            # ----------------------------------------------------------
            # PRIMARY CRAWLER CELL
            # ----------------------------------------------------------

            cells = [
                {
                    "row": row,
                    "col": col,
                    "source": position["direction"],
                    "k": position["k"],
                    "perturbation_k": position[
                        "perturbation_k"
                    ],
                }
            ]

            # ----------------------------------------------------------
            # ELASTIC CLOUD
            # ----------------------------------------------------------

            if (
                L is not None
                and S is not None
                and c is not None
            ):

                elastic = self.elastic_candidates(
                    L=L,
                    S=S,
                    c=c,
                )

                for cloud_cell in elastic:

                    cells.append({
                        "row": cloud_cell["row"],
                        "col": cloud_cell["col"],
                        "source": "elastic",
                        "k": position["k"],
                        "perturbation_k": position[
                            "perturbation_k"
                        ],
                    })

            # ----------------------------------------------------------
            # READ ALL CANDIDATE CELLS
            # ----------------------------------------------------------

            seen_cells: Set[
                Tuple[int, int]
            ] = set()

            for cell in cells:

                cell_key = (
                    int(cell["row"]),
                    int(cell["col"]),
                )

                if cell_key in seen_cells:
                    continue

                seen_cells.add(
                    cell_key
                )

                entries = self.crawl_cell(
                    cell["row"],
                    cell["col"],
                )

                for entry in entries:

                    key = _candidate_key(
                        entry
                    )

                    if key in seen_candidates:
                        continue

                    seen_candidates.add(
                        key
                    )

                    result = dict(
                        entry
                    )

                    result["crawl"] = {
                        "row": cell["row"],
                        "col": cell["col"],
                        "source": cell["source"],
                        "k": cell["k"],
                        "perturbation_k": cell[
                            "perturbation_k"
                        ],
                    }

                    candidates.append(
                        result
                    )

                    if len(
                        candidates
                    ) >= limit:

                        return candidates

        return candidates

    # ========================================================================
    # WORD-BASED CRAWL
    # ========================================================================

    def crawl_word(
        self,
        word: str,
        lang: str = "en",
        limit: int = DEFAULT_LIMIT,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Enter the crawler through a word.

        keyboard.py calculates:

            Lsum
            Ssum
            first-letter index
            canonical GSP start_row

        The crawler then performs K/D traversal.
        """

        normalized = normalise(
            word,
            lang,
        )

        Lsum = calculate_lsum(
            normalized,
            lang,
        )

        Ssum = calculate_ssum(
            normalized,
            lang,
        )

        c = first_letter_index(
            normalized,
            lang,
        )

        start_row = self.canonical_start_row(
            normalized,
            lang,
        )

        return self.crawl(
            start_row=start_row,
            start_col=c,
            L=Lsum,
            S=Ssum,
            c=c,
            limit=limit,
        )

    # ========================================================================
    # EXTERNAL CANDIDATE INDEXING
    # ========================================================================

    def index_crawled_candidate(
        self,
        text: str,
        lang: str = "en",
        source: str = "crawler",
    ) -> Optional[int]:
        """
        Index crawled external information into MemoryGrid.

        MemoryGrid remains responsible for tokenization and indexing.

        The crawler does not duplicate MemoryGrid's storage logic.
        """

        if not text:
            return None

        if not hasattr(
            self.memory,
            "add_document",
        ):
            return None

        return self.memory.add_document(
            text=text,
            lang=lang,
            source=source,
        )

    # ========================================================================
    # INDEX MANY CRAWLED CANDIDATES
    # ========================================================================

    def index_crawled_candidate