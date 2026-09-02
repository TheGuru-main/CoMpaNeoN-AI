"""
CoMpaNeoN Memory Partition
==========================

Memory partitioning and contextual memory routing layer.

ARCHITECTURE
------------

                        MemoryGrid
                            │
                            ▼
                    MemoryPartition
                            │
             ┌──────────────┼──────────────┐
             │              │              │
          Domain         Project        Context
             │              │              │
             ▼              ▼              ▼
         Hierarchy       Trace/Pins     Relevancy
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                      GSP XOR Quorum
                            │
                            ▼
                       GridCV intact
                            │
                            ▼
                    Partition Retrieval
                            │
                            ▼
                 Higher Brain Constituents


RESPONSIBILITIES
----------------

MemoryPartition owns:

    - partition construction
    - deterministic partition identity
    - GSP XOR quorum sharding
    - domain partitioning
    - hierarchy partitioning
    - role partitioning
    - relevancy partitioning
    - project traces
    - project pins
    - project iteration
    - project state inspection
    - list state
    - any state
    - while-loop state
    - state comparison
    - cross-project comparison
    - project-context awareness
    - partition retrieval
    - GridCV validation/comparison

MemoryPartition does NOT own:

    - tokenization
    - language mathematics
    - alphabet mathematics
    - word placement
    - GSP linguistic placement
    - crawler traversal
    - external acquisition
    - ranking
    - prompt generation
    - rules
    - tool execution
    - AI response generation

AUTHORITIES
-----------

tokenizer.py
    Linguistic authority.

keyboard.py / placement.py
    Canonical GSP placement authority.

MemoryGrid
    Canonical storage/indexing authority.

GridCrawler
    Grid traversal/router authority.

WebCrawler
    External knowledge acquisition authority.

GridCV
    Grid comparison/validation authority.

MemoryPartition
    Memory partitioning and contextual sharding authority.
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass, field

from datetime import datetime, timezone

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)


# ============================================================================
# GRIDCV
# ============================================================================

try:
    from grid_cv import GridCV

except ImportError:

    GridCV = None


# ============================================================================
# MEMORYGRID
# ============================================================================

try:
    from memory_grid import MemoryGrid

except ImportError:

    MemoryGrid = Any


# ============================================================================
# GRID CRAWLER
# ============================================================================

try:
    from grid_crawler import GridCrawler

except ImportError:

    GridCrawler = None


# ============================================================================
# CRAWLER RETRIEVAL
# ============================================================================

try:
    from crawler_retrieval import CrawlerRetrieval

except ImportError:

    CrawlerRetrieval = None


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ROWS = 64

DEFAULT_COLS = 46

DEFAULT_RELEVANCY = 0.0

DEFAULT_ROLE = "general"

DEFAULT_DOMAIN = "general"

DEFAULT_HIERARCHY = "root"

DEFAULT_PROJECT_STATE = "active"

DEFAULT_PARTITION_LIMIT = 250


# ============================================================================
# TIME
# ============================================================================

def utc_now() -> str:
    """
    Return current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_text(
    value: Any,
) -> str:
    """
    Normalize arbitrary values into deterministic strings.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


# ============================================================================
# STABLE HASH
# ============================================================================

def stable_hash(
    value: Any,
) -> str:
    """
    Create deterministic SHA-256 identity.
    """

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):

        try:

            value = json.dumps(
                value,
                sort_keys=True,
                default=str,
            )

        except Exception:

            value = str(
                value
            )

    return hashlib.sha256(
        str(value)
        .encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================================
# INTEGER HASH
# ============================================================================

def stable_int(
    value: Any,
) -> int:
    """
    Deterministic integer representation.
    """

    digest = stable_hash(
        value
    )

    return int(
        digest[:16],
        16,
    )


# ============================================================================
# XOR
# ============================================================================

def xor_values(
    *values: int,
) -> int:
    """
    XOR arbitrary integer values.
    """

    result = 0

    for value in values:

        result ^= int(
            value
        )

    return result


# ============================================================================
# QUORUM
# ============================================================================

def quorum_value(
    values: Iterable[int],
) -> int:
    """
    Deterministic quorum value.

    The quorum is formed from the XOR of
    all participating values.
    """

    result = 0

    for value in values:

        result ^= int(
            value
        )

    return result


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class PartitionKey:
    """
    Canonical partition identity.
    """

    domain: str = DEFAULT_DOMAIN

    hierarchy: str = DEFAULT_HIERARCHY

    role: str = DEFAULT_ROLE

    project_id: str = ""

    project_context: str = ""

    relevancy_bucket: int = 0


@dataclass
class ProjectTrace:
    """
    Persistent project trace.

    Tracks how knowledge enters and evolves
    within a project.
    """

    project_id: str

    trace_id: str

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    events: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )


@dataclass
class ProjectPin:
    """
    Explicit persistent project memory pin.
    """

    project_id: str

    pin_id: str

    value: Any

    created_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


@dataclass
class ProjectIteration:
    """
    Tracks iterative project development.
    """

    project_id: str

    iteration: int

    state: str

    created_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================================
# MEMORY PARTITION
# ============================================================================

class MemoryPartition:
    """
    CoMpaNeoN memory partitioning layer.

    Memory enters through MemoryGrid.

    This layer does not replace MemoryGrid.

    It creates deterministic contextual
    partitions around already-indexed
    knowledge and memory.
    """

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        memory_grid: MemoryGrid,
        grid_cv: Optional[
            Any
        ] = None,
        grid_crawler: Optional[
            Any
        ] = None,
        crawler_retrieval: Optional[
            Any
        ] = None,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
    ) -> None:

        self.memory = (
            memory_grid
        )

        self.rows = int(
            getattr(
                memory_grid,
                "rows",
                rows,
            )
        )

        self.cols = int(
            getattr(
                memory_grid,
                "cols",
                cols,
            )
        )

        # --------------------------------------------------------------
        # GridCV
        # --------------------------------------------------------------

        if grid_cv is not None:

            self.grid_cv = (
                grid_cv
            )

        elif GridCV is not None:

            try:

                self.grid_cv = (
                    GridCV()
                )

            except Exception:

                self.grid_cv = (
                    None
                )

        else:

            self.grid_cv = (
                None
            )

        # --------------------------------------------------------------
        # GridCrawler
        # --------------------------------------------------------------

        if grid_crawler is not None:

            self.grid_crawler = (
                grid_crawler
            )

        elif GridCrawler is not None:

            try:

                self.grid_crawler = (
                    GridCrawler(
                        memory_grid
                    )
                )

            except Exception:

                self.grid_crawler = (
                    None
                )

        else:

            self.grid_crawler = (
                None
            )

        # --------------------------------------------------------------
        # Crawler Retrieval
        # --------------------------------------------------------------

        if crawler_retrieval is not None:

            self.crawler_retrieval = (
                crawler_retrieval
            )

        elif CrawlerRetrieval is not None:

            try:

                self.crawler_retrieval = (
                    CrawlerRetrieval(
                        memory_grid
                    )
                )

            except Exception:

                self.crawler_retrieval = (
                    None
                )

        else:

            self.crawler_retrieval = (
                None
            )

        # --------------------------------------------------------------
        # PARTITION STORAGE
        # --------------------------------------------------------------

        self.partitions: Dict[
            str,
            List[
                Dict[str, Any]
            ],
        ] = {}

        self.partition_metadata: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # --------------------------------------------------------------
        # PROJECTS
        # --------------------------------------------------------------

        self.project_traces: Dict[
            str,
            ProjectTrace,
        ] = {}

        self.project_pins: Dict[
            str,
            Dict[
                str,
                ProjectPin,
            ],
        ] = {}

        self.project_iterations: Dict[
            str,
            List[
                ProjectIteration
            ],
        ] = {}

        self.project_states: Dict[
            str,
            Dict[
                str,
                Any,
            ],
        ] = {}

    # ======================================================================
    # GRID DIMENSIONS
    # ======================================================================

    def grid_dimensions(
        self,
    ) -> Tuple[
        int,
        int,
    ]:
        """
        Return MemoryGrid dimensions.

        Current architecture:

            64 rows
            46 columns

        The partition layer follows
        MemoryGrid dimensions rather
        than creating its own grid.
        """

        return (
            self.rows,
            self.cols,
        )

    # ======================================================================
    # RELEVANCY BUCKET
    # ======================================================================

    def relevancy_bucket(
        self,
        relevancy: float,
        buckets: int = 10,
    ) -> int:
        """
        Convert relevancy into
        deterministic partition bucket.
        """

        value = float(
            relevancy
        )

        value = max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

        return min(
            buckets - 1,
            int(
                value * buckets
            ),
        )

    # ======================================================================
    # BUILD PARTITION KEY
    # ======================================================================

    def build_partition_key(
        self,
        *,
        domain: str = DEFAULT_DOMAIN,
        hierarchy: str = DEFAULT_HIERARCHY,
        role: str = DEFAULT_ROLE,
        project_id: str = "",
        project_context: str = "",
        relevancy: float = (
            DEFAULT_RELEVANCY
        ),
    ) -> PartitionKey:
        """
        Build canonical contextual partition key.
        """

        return PartitionKey(
            domain=(
                normalize_text(
                    domain
                )
                or DEFAULT_DOMAIN
            ),

            hierarchy=(
                normalize_text(
                    hierarchy
                )
                or DEFAULT_HIERARCHY
            ),

            role=(
                normalize_text(
                    role
                )
                or DEFAULT_ROLE
            ),

            project_id=(
                normalize_text(
                    project_id
                )
            ),

            project_context=(
                normalize_text(
                    project_context
                )
            ),

            relevancy_bucket=(
                self.relevancy_bucket(
                    relevancy
                )
            ),
        )

    # ======================================================================
    # PARTITION ID
    # ======================================================================

    def partition_id(
        self,
        key: PartitionKey,
    ) -> str:
        """
        Deterministic partition identifier.
        """

        payload = {

            "domain":
                key.domain,

            "hierarchy":
                key.hierarchy,

            "role":
                key.role,

            "project_id":
                key.project_id,

            "project_context":
                key.project_context,

            "relevancy_bucket":
                key.relevancy_bucket,

        }

        return stable_hash(
            payload
        )

    # ======================================================================
    # GSP XOR QUORUM
    # ======================================================================

    def gsp_xor_quorum(
        self,
        *,
        domain: str,
        hierarchy: str,
        role: str,
        relevancy_bucket: int,
        project_id: str = "",
        project_context: str = "",
    ) -> Dict[
        str,
        int,
    ]:
        """
        Produce deterministic shard routing
        from contextual dimensions.

        This does not replace GSP placement.

        It determines which contextual
        memory partition/shard should own
        the record after MemoryGrid has
        handled canonical storage.

        Quorum dimensions:

            domain
            hierarchy
            role
            relevancy
            project
            project context

        The dimensions are XORed into a
        deterministic quorum value.
        """

        domain_value = stable_int(
            domain
        )

        hierarchy_value = stable_int(
            hierarchy
        )

        role_value = stable_int(
            role
        )

        relevancy_value = int(
            relevancy_bucket
        )

        project_value = stable_int(
            project_id
        )

        context_value = stable_int(
            project_context
        )

        quorum = quorum_value(
            [
                domain_value,
                hierarchy_value,
                role_value,
                relevancy_value,
                project_value,
                context_value,
            ]
        )

        xor = xor_values(
            domain_value,
            hierarchy_value,
            role_value,
            relevancy_value,
            project_value,
            context_value,
        )

        rows, cols = (
            self.grid_dimensions()
        )

        shard_row = (
            quorum % rows
        ) + 1

        shard_col = (
            xor % cols
        )

        return {

            "quorum":
                quorum,

            "xor":
                xor,

            "row":
                shard_row,

            "col":
                shard_col,

        }

    # ======================================================================
    # PROJECT TRACE
    # ======================================================================

    def ensure_project_trace(
        self,
        project_id: str,
    ) -> ProjectTrace:
        """
        Ensure a project trace exists.
        """

        project_id = (
            normalize_text(
                project_id
            )
        )

        if project_id not in (
            self.project_traces
        ):

            trace = ProjectTrace(

                project_id=
                    project_id,

                trace_id=
                    stable_hash(
                        {
                            "project":
                                project_id
                        }
                    ),

            )

            self.project_traces[
                project_id
            ] = trace

        return self.project_traces[
            project_id
        ]

    # ======================================================================
    # ADD PROJECT TRACE EVENT
    # ======================================================================

    def trace_project(
        self,
        project_id: str,
        event: Dict[
            str,
            Any,
        ],
    ) -> ProjectTrace:
        """
        Append deterministic project trace event.
        """

        trace = (
            self.ensure_project_trace(
                project_id
            )
        )

        trace.events.append({

            "timestamp":
                utc_now(),

            "eve