"""
CoMpaNeoN Memory Partition
==========================

Memory partition layer for CoMpaNeoN.

ARCHITECTURE
------------

    tokenizer.py
         │
         ▼
    MemoryGrid
         │
         ├── GridCrawler
         │
         ├── CrawlerRetrieval
         │
         ▼
    MemoryPartition
         │
         ├── GSP partition identity
         ├── XOR partition routing
         ├── Quorum validation
         ├── Domain partitioning
         ├── Hierarchy partitioning
         ├── Role partitioning
         ├── Relevancy partitioning
         ├── Project trace
         ├── Project pin
         ├── Project iteration
         ├── Compare state
         ├── List state
         ├── Any state
         ├── While-loop state
         ├── Cross-project comparison
         └── Project-context-aware retrieval

         │
         ▼

    Linguistic / Semantic Layer

IMPORTANT
---------

MemoryPartition does NOT own:

    - tokenization
    - language alphabet mathematics
    - 46-column linguistic placement
    - MemoryGrid document placement
    - canonical GSP start-row calculation
    - crawler K/D traversal mathematics
    - Elastic Cloud mathematics
    - final AI response generation

Those remain owned by their canonical layers.

MemoryPartition owns MEMORY CONTEXT PARTITIONING.

The partition layer determines how already-indexed memory is grouped,
traced, routed, validated, compared and supplied to higher layers.

GRID
----

MemoryGrid remains:

    46 columns
    64 rows

GridCrawler remains the router through the memory grid.

CrawlerRetrieval remains responsible for retrieval orchestration and
ranking.

MemoryPartition sits above those layers and creates structured memory
contexts.

GSP + XOR + QUORUM
------------------

GSP:
    deterministic memory routing identity.

XOR:
    creates a deterministic partition signature from multiple context
    dimensions.

QUORUM:
    validates that enough contextual dimensions agree before a partition
    is considered a strong contextual match.

GridCV:
    partition/context validation representation.

"""

from __future__ import annotations

import hashlib

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    datetime,
    timezone,
)

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
# OPTIONAL ARCHITECTURE IMPORTS
# ============================================================================

try:

    from grid_crawler import GridCrawler

except ImportError:

    GridCrawler = None  # type: ignore


try:

    from crawler_retrieval import CrawlerRetrieval

except ImportError:

    CrawlerRetrieval = None  # type: ignore


# ============================================================================
# PARTITION CONSTANTS
# ============================================================================

DEFAULT_PARTITION_LIMIT = 250

DEFAULT_QUORUM = 2

MAX_ITERATION_DEPTH = 64

PROJECT_TRACE_LIMIT = 512

GRIDCV_VECTOR_SIZE = 64


# ============================================================================
# TIME
# ============================================================================

def _timestamp() -> str:
    """
    UTC timestamp used for partition tracing.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================================
# STABLE HASH
# ============================================================================

def _stable_hash(
    value: Any,
) -> int:
    """
    Deterministic integer hash.

    Python's built-in hash is intentionally not used because it may vary
    between processes.

    Memory partition routing must remain deterministic.
    """

    encoded = (
        str(value)
        .encode(
            "utf-8"
        )
    )

    digest = hashlib.sha256(
        encoded
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


# ============================================================================
# NORMALIZATION
# ============================================================================

def _normalise_key(
    value: Any,
) -> str:
    """
    Normalize partition labels.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


# ============================================================================
# LIST NORMALIZATION
# ============================================================================

def _as_list(
    value: Any,
) -> List[Any]:
    """
    Convert a value into a list without treating strings as iterables.
    """

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(
            value
        )

    if isinstance(
        value,
        set,
    ):
        return list(
            value
        )

    return [
        value
    ]


# ============================================================================
# UNIQUE PRESERVING ORDER
# ============================================================================

def _unique(
    values: Iterable[Any],
) -> List[Any]:
    """
    Remove duplicates while preserving order.
    """

    result = []

    seen: Set[
        str
    ] = set()

    for value in values:

        key = repr(
            value
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            value
        )

    return result


# ============================================================================
# GRIDCV
# ============================================================================

@dataclass
class GridCV:
    """
    Grid Context Vector.

    GridCV remains a context-validation layer.

    It does NOT replace:

        - GridCrawler
        - MemoryGrid
        - GSP
        - crawler traversal

    GridCV represents partition dimensions as a deterministic context
    vector which can be compared with another partition.
    """

    domain: str = ""

    hierarchy: str = ""

    role: str = ""

    relevancy: str = ""

    project_id: str = ""

    project_trace: List[
        str
    ] = field(
        default_factory=list
    )

    values: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------------

    def vector(
        self,
        size: int = GRIDCV_VECTOR_SIZE,
    ) -> List[int]:
        """
        Build a deterministic GridCV vector.

        The vector is derived from contextual dimensions, not from
        linguistic alphabet placement.
        """

        size = max(
            1,
            int(size),
        )

        vector = [
            0
        ] * size

        components = {

            "domain":
                self.domain,

            "hierarchy":
                self.hierarchy,

            "role":
                self.role,

            "relevancy":
                self.relevancy,

            "project_id":
                self.project_id,

        }

        for key, value in components.items():

            if not value:
                continue

            hashed = _stable_hash(
                f"{key}:{value}"
            )

            index = (
                hashed
                % size
            )

            vector[
                index
            ] += 1

        for trace in self.project_trace:

            hashed = _stable_hash(
                f"trace:{trace}"
            )

            index = (
                hashed
                % size
            )

            vector[
                index
            ] += 1

        for key, value in sorted(
            self.values.items()
        ):

            hashed = _stable_hash(
                f"{key}:{value}"
            )

            index = (
                hashed
                % size
            )

            vector[
                index
            ] += 1

        return vector

    # ------------------------------------------------------------------------

    def similarity(
        self,
        other: "GridCV",
    ) -> float:
        """
        Compare two GridCV vectors using intersection-over-union style
        vector agreement.

        Returns:

            0.0 -> no agreement
            1.0 -> complete agreement
        """

        left = self.vector()

        right = other.vector()

        intersection = 0

        union = 0

        for a, b in zip(
            left,
            right,
        ):

            intersection += min(
                a,
                b,
            )

            union += max(
                a,
                b,
            )

        if union == 0:
            return 0.0

        return (
            intersection
            / union
        )

    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        return {

            "domain":
                self.domain,

            "hierarchy":
                self.hierarchy,

            "role":
                self.role,

            "relevancy":
                self.relevancy,

            "project_id":
                self.project_id,

            "project_trace":
                list(
                    self.project_trace
                ),

            "values":
                dict(
                    self.values
                ),

        }


# ============================================================================
# PROJECT TRACE
# ============================================================================

@dataclass
class ProjectTrace:
    """
    Tracks memory movement through project contexts.

    ProjectTrace preserves:

        where the memory originated,
        which project contexts consumed it,
        how it moved,
        and which partition states were involved.
    """

    project_id: str

    events: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------------

    def add(
        self,
        event: str,
        **data: Any,
    ) -> Dict[
        str,
        Any
    ]:

        record = {

            "event":
                _normalise_key(
                    event
                ),

            "timestamp":
                _timestamp(),

            "data":
                dict(
                    data
                ),

        }

        self.events.append(
            record
        )

        if len(
            self.events
        ) > PROJECT_TRACE_LIMIT:

            self.events = (
                self.events[
                    -PROJECT_TRACE_LIMIT:
                ]
            )

        return record

    # ------------------------------------------------------------------------

    def names(
        self,
    ) -> List[
        str
    ]:
        """
        Return event names for GridCV context.
        """

        return [

            event.get(
                "event",
                "",
            )

            for event in self.events

        ]

    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        return {

            "project_id":
                self.project_id,

            "events":
                list(
                    self.events
                ),

        }


# ============================================================================
# PROJECT PIN
# ============================================================================

@dataclass
class ProjectPin:
    """
    Persistent contextual anchor.

    A pin is not ordinary memory ranking.

    It explicitly tells the partition layer that a memory/project context
    must remain available during project iteration.
    """

    project_id: str

    key: str

    value: Any

    created_at: str = field(
        default_factory=_timestamp
    )

    updated_at: str = field(
        default_factory=_timestamp
    )

    active: bool = True

    # ------------------------------------------------------------------------

    def update(
        self,
        value: Any,
    ) -> None:

        self.value = value

        self.updated_at = (
            _timestamp()
        )

        self.active = True

    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        return {

            "project_id":
                self.project_id,

            "key":
                self.key,

            "value":
                self.value,

            "created_at":
                self.created_at,

            "updated_at":
                self.updated_at,

            "active":
                self.active,

        }


# ============================================================================
# PARTITION STATE
# ============================================================================

@dataclass
class PartitionState:
    """
    Stateful project/context representation.

    Explicitly supports the architecture states discussed:

        list
        any
        while_loop
        compare
        iteration
    """

    project_id: str

    list_state: List[
        Any
    ] = field(
        default_factory=list
    )

    any_state: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    while_state: Dict[
        str,
        Dict[str, Any]
    ] = field(
        default_factory=dict
    )

    compare_state: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    iteration: int = 0

    updated_at: str = field(
        default_factory=_timestamp
    )

    # ------------------------------------------------------------------------

    def add_list(
        self,
        value: Any,
    ) -> None:

        self.list_state.append(
            value
        )

        self.updated_at = (
            _timestamp()
        )

    # ------------------------------------------------------------------------

    def set_any(
        self,
        key: str,
        value: Any,
    ) -> None:

        self.any_state[
            str(key)
        ] = value

        self.updated_at = (
            _timestamp()
        )

    # ------------------------------------------------------------------------

    def set_while(
        self,
        key: str,
        *,
        active: bool = True,
        value: Any = None,
        iterations: int = 0,
    ) -> None:

        self.while_state[
            str(key)
        ] = {

            "active":
                bool(active),

            "value":
                value,

            "iterations":
                int(iterations),

            "updated_at":
                _timestamp(),

        }

        self.updated_at = (
            _timestamp()
        )

    # ------------------------------------------------------------------------

    def increment_iteration(
        self,
    ) -> int:

        self.iteration += 1

        self.updated_at = (
            _timestamp()
        )

        return self.iteration

    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        return {

            "project_id":
                self.project_id,

            "list":
                list(
                    self.list_state
                ),

            "any":
                dict(
                    self.any_state
                ),

            "while":
                dict(
                    self.while_state
                ),

            "compare":
                dict(
                    self.compare_state
                ),

            "iteration":
                self.iteration,

            "updated_at":
                self.updated_at,

        }


# ============================================================================
# MEMORY PARTITION RECORD
# ============================================================================

@dataclass
class MemoryPartitionRecord:
    """
    One contextual partition over MemoryGrid candidates.
    """

    partition_id: str

    xor_signature: int

    gsp_identity: Dict[
        str,
        Any
    ]

    domain: str

    hierarchy: str

    role: str

    relevancy: str

    project_id: str

    document_ids: List[
        Any
    ] = field(
        default_factory=list
    )

    candidates: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    gridcv: Optional[
        GridCV
    ] = None

    trace: Optional[
        ProjectTrace
    ] = None

    state: Optional[
        PartitionState
    ] = None

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=_timestamp
    )

    updated_at: str = field(
        default_factory=_timestamp
    )

    # ------------------------------------------------------------------------

    def add_candidate(
        self,
        candidate: Dict[
            str,
            Any
        ],
    ) -> None:

        self.candidates.append(
            dict(
                candidate
            )
        )

        doc_id = candidate.get(
            "doc_id"
        )

        if (
            doc_id is not None
            and doc_id not in self.document_ids
        ):

            self.document_ids.append(
                doc_id
            )

        self.updated_at = (
            _timestamp()
        )

    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        return {

            "partition_id":
                self.partition_id,

            "xor_signature":
                self.xor_signature,

            "gsp_identity":
                dict(
                    self.gsp_identity
                ),

            "domain":
                self.domain,

            "hierarchy":
                self.hierarchy,

            "role":
                self.role,

            "relevancy":
                self.relevancy,

            "project_id":
                self.project_id,

            "document_ids":
                list(
                    self.document_ids
                ),

            "candidate_count":
                len(
                    self.candidates
                ),

            "gridcv":
                (
                    self.gridcv.to_dict()
                    if self.gridcv
                    else None
                ),

            "trace":
                (
                    self.trace.to_dict()
                    if self.trace
                    else None
                ),

            "state":
                (
                    self.state.to_dict()
                    if self.state
                    else None
                ),

            "metadata":
                dict(
                    self.metadata
                ),

            "created_at":
                self.created_at,

            "updated_at":
                self.updated_at,

        }


# ============================================================================
# MEMORY PARTITION
# ============================================================================

class MemoryPartition:
    """
    CoMpaNeoN contextual memory partition layer.

    The class receives the existing MemoryGrid and crawler architecture.
