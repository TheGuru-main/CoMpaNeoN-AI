"""
CoMpaNeoN GridCV
================

GridCV is the canonical validation and vector layer for
CoMpaNeoN brain partitions.

Architecture
------------

tokenizer.py
    ↓
placement.py
    ↓
memory_grid.py
    ↓
grid_crawler.py / web_crawler.py
    ↓
crawler_retrieval.py
    ↓
memory_partition.py
    ├── GSP shard routing
    ├── XOR shard routing
    ├── quorum
    ├── domain
    ├── hierarchy
    ├── role
    ├── relevancy
    ├── project trace
    ├── project pin
    ├── project iteration
    ├── state
    ├── project comparison
    └── project-context-aware
            ↓
        grid_cv.py
            ├── partition validation
            ├── deterministic vectors
            ├── partition signatures
            ├── comparison vectors
            ├── state vectors
            └── relationship-compatible signals
            ↓
relation_and_alphabet_matrix.py
            ↓
linguistic / semantic layers
            ↓
brain constituents
            ↓
AI response


RESPONSIBILITIES
----------------

GridCV:

    - validates partition structures
    - preserves partition metadata
    - creates deterministic vectors
    - creates stable partition signatures
    - represents domain information
    - represents hierarchy information
    - represents role information
    - represents relevancy information
    - represents project traces
    - represents project pins
    - represents project iterations
    - represents project state
    - represents list / any / while-loop state
    - compares partitions
    - compares projects
    - supports cross-project comparison
    - produces project-context-aware vectors
    - exposes signals for relationship matrix analysis

GridCV does NOT own:

    - tokenization
    - language detection
    - linguistic placement
    - GSP mathematics
    - crawler traversal
    - Elastic Cloud
    - external crawling
    - document indexing
    - MemoryGrid storage
    - shard selection authority
    - ranking
    - prompt generation
    - tool execution
    - AI response generation


IMPORTANT
---------

memory_partition.py remains the owner of partition routing.

GridCV does not independently decide where memory belongs.

It validates and represents the partition result.

This prevents GridCV from contradicting:

    tokenizer.py
    placement.py
    memory_grid.py
    grid_crawler.py
    web_crawler.py
    memory_partition.py
"""

from __future__ import annotations

import hashlib
import json
import math

from collections import Counter
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

GRID_CV_VERSION = "1.0"

DEFAULT_VECTOR_SIZE = 64

DEFAULT_COMPARE_LIMIT = 64


# ============================================================================
# CANONICAL PARTITION FIELDS
# ============================================================================

PARTITION_FIELDS = (
    "domain",
    "hierarchy",
    "role",
    "relevancy",
    "project_trace",
    "project_pin",
    "project_iteration",
    "state",
    "project_context_aware",
)


# ============================================================================
# HELPERS
# ============================================================================

def _stable_json(
    value: Any,
) -> str:
    """
    Convert arbitrary supported data into a deterministic string.
    """

    try:

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    except Exception:

        return str(
            value
        )


def _stable_hash(
    value: Any,
) -> str:
    """
    SHA-256 deterministic identity.

    GridCV uses hashes for vector generation and signatures.

    This is not a replacement for GSP/XOR partition routing.
    """

    payload = _stable_json(
        value
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def _safe_list(
    value: Any,
) -> List[Any]:
    """
    Normalize a value into a list.
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

        return sorted(
            value,
            key=str,
        )

    return [
        value
    ]


def _safe_dict(
    value: Any,
) -> Dict[str, Any]:
    """
    Normalize mapping-like data.
    """

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


def _normalise_text(
    value: Any,
) -> str:

    return (
        str(
            value
        )
        .strip()
        .lower()
    )


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(
                value
            ),
        ),
    )


# ============================================================================
# DETERMINISTIC VECTOR
# ============================================================================

def deterministic_vector(
    value: Any,
    size: int = DEFAULT_VECTOR_SIZE,
) -> List[float]:
    """
    Convert a value into a deterministic vector.

    No randomness is used.

    The same value always produces the same vector.
    """

    size = max(
        1,
        int(size),
    )

    digest = bytes.fromhex(
        _stable_hash(
            value
        )
    )

    vector = []

    for index in range(
        size
    ):

        byte = digest[
            index % len(
                digest
            )
        ]

        vector.append(
            round(
                byte / 255.0,
                8,
            )
        )

    return vector


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def vector_magnitude(
    vector: Sequence[
        float
    ],
) -> float:

    return math.sqrt(
        sum(
            float(value)
            * float(value)
            for value in vector
        )
    )


def normalize_vector(
    vector: Sequence[
        float
    ],
) -> List[float]:

    magnitude = vector_magnitude(
        vector
    )

    if magnitude == 0:

        return [
            0.0
            for _ in vector
        ]

    return [
        float(value)
        / magnitude
        for value in vector
    ]


def cosine_similarity(
    vector_a: Sequence[
        float
    ],
    vector_b: Sequence[
        float
    ],
) -> float:
    """
    Return cosine similarity in the range:

        0.0 - 100.0
    """

    if not vector_a:
        return 0.0

    if not vector_b:
        return 0.0

    size = min(
        len(vector_a),
        len(vector_b),
    )

    if size == 0:

        return 0.0

    a = list(
        vector_a[:size]
    )

    b = list(
        vector_b[:size]
    )

    magnitude_a = vector_magnitude(
        a
    )

    magnitude_b = vector_magnitude(
        b
    )

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):

        return 0.0

    dot = sum(
        float(a[index])
        * float(b[index])
        for index in range(
            size
        )
    )

    similarity = dot / (
        magnitude_a
        * magnitude_b
    )

    return round(
        _clamp(
            similarity,
            -1.0,
            1.0,
        )
        * 100.0,
        8,
    )


def vector_average(
    vectors: Sequence[
        Sequence[float]
    ],
) -> List[float]:

    vectors = [
        list(vector)
        for vector in vectors
        if vector
    ]

    if not vectors:

        return []

    size = min(
        len(vector)
        for vector in vectors
    )

    if size == 0:

        return []

    result = []

    for index in range(
        size
    ):

        value = sum(
            float(vector[index])
            for vector in vectors
        )

        result.append(
            value
            / len(vectors)
        )

    return result


# ============================================================================
# PARTITION VALIDATION
# ============================================================================

class GridCV:
    """
    CoMpaNeoN Grid Cross-Validation and Vector layer.

    GridCV receives partition information.

    It does not independently create partition routing.
    """

    def __init__(
        self,
        vector_size: int = (
            DEFAULT_VECTOR_SIZE
        ),
    ) -> None:

        self.vector_size = max(
            1,
            int(
                vector_size
            ),
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_partition(
        self,
        partition: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        """
        Validate a memory partition.

        Validation is structural.

        Routing remains the responsibility of memory_partition.py.
        """

        if not isinstance(
            partition,
            Mapping,
        ):

            return {
                "valid": False,
                "errors": [
                    "partition must be a mapping"
                ],
                "missing": list(
                    PARTITION_FIELDS
                ),
                "present": [],
            }

        errors = []

        present = []

        missing = []

        for field in PARTITION_FIELDS:

            if field in partition:

                present.append(
                    field
                )

            else:

                missing.append(
                    field
                )

        # --------------------------------------------------------------------
        # GSP ROUTING
        # --------------------------------------------------------------------

        if (
            "gsp"
            not in partition
            and
            "gsp_shard"
            not in partition
            and
            "start_row"
            not in partition
        ):

            errors.append(
                "missing GSP partition routing metadata"
            )

        # --------------------------------------------------------------------
        # XOR ROUTING
        # --------------------------------------------------------------------

        if (
            "xor"
            not in partition
            and
            "xor_shard"
            not in partition
        ):

            errors.append(
                "missing XOR partition routing metadata"
            )

        # --------------------------------------------------------------------
        # QUORUM
        # --------------------------------------------------------------------

        if (
            "quorum"
            not in partition
        ):

            errors.append(
                "missing quorum metadata"
            )

        return {
            "valid": (
                len(errors) == 0
            ),
            "errors": errors,
            "missing": missing,
            "present": present,
        }

    # ========================================================================
    # PARTITION SIGNATURE
    # ========================================================================

    def partition_signature(
        self,
        partition: Mapping[
            str,
            Any,
        ],
    ) -> str:
        """
        Stable partition identity.

        This identifies the resulting partition state.

        It does not replace GSP/XOR identity.
        """

        payload = {
            key: partition.get(
                key
            )
            for key in (
                "gsp",
                "gsp_shard",
                "start_row",
                "xor",
                "xor_shard",
                "quorum",
                "domain",
                "hierarchy",
                "role",
                "relevancy",
                "project_trace",
                "project_pin",
                "project_iteration",
                "state",
                "project_context_aware",
            )
        }

        return _stable_hash(
            payload
        )

    # ========================================================================
    # DOMAIN VECTOR
    # ========================================================================

    def domain_vector(
        self,
        domain: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "domain": domain,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # HIERARCHY VECTOR
    # ========================================================================

    def hierarchy_vector(
        self,
        hierarchy: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "hierarchy": hierarchy,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # ROLE VECTOR
    # ========================================================================

    def role_vector(
        self,
        role: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "role": role,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # RELEVANCY VECTOR
    # ========================================================================

    def relevancy_vector(
        self,
        relevancy: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "relevancy": relevancy,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # PROJECT TRACE VECTOR
    # ========================================================================

    def project_trace_vector(
        self,
        trace: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "project_trace": trace,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # PROJECT PIN VECTOR
    # ========================================================================

    def project_pin_vector(
        self,
        pin: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "project_pin": pin,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # PROJECT ITERATION VECTOR
    # ========================================================================

    def project_iteration_vector(
        self,
        iteration: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "project_iteration": iteration,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # STATE SIGNAL
    # ========================================================================

    def state_signal(
        self,
        state: Any,
    ) -> Dict[str, Any]:
        """
        Represent project execution state.

        Includes:

            list state
            any state
            while-loop state

        These are represented as state information.

        GridCV does not execute arbitrary loops.
        """

        state_data = _safe_dict(
            state
        )

        values = _safe_list(
            state_data.get(
                "list",
                state_data.get(
                    "items",
                    [],
                ),
            )
        )

        any_value = state_data.get(
            "any"
        )

        while_state = state_data.get(
            "while"
        )

        completed = state_data.get(
            "completed"
        )

        active = state_data.get(
            "active"
        )

        return {
            "list": values,
            "list_count": len(
                values
            ),
            "any": any_value,
            "any_truthy": bool(
                any_value
            ),
            "while": while_state,
            "while_active": bool(
                while_state
            ),
            "completed": completed,
            "active": active,
        }

    # ========================================================================
    # STATE VECTOR
    # ========================================================================

    def state_vector(
        self,
        state: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "state": self.state_signal(
                    state
                ),
            },
            size=self.vector_size,
        )

    # ========================================================================
    # PROJECT CONTEXT VECTOR
    # ========================================================================

    def project_context_vector(
        self,
        context: Any,
    ) -> List[float]:

        return deterministic_vector(
            {
                "project_context_aware": context,
            },
            size=self.vector_size,
        )

    # ========================================================================
    # ROUTING VECTOR
    # ========================================================================

    def routing_vector(
        self,
        partition: Mapping[
            str,
            Any,
        ],
    ) -> List[float]:
        """
        Represent routing metadata without taking ownership of routing.
        """

        routing = {
            "gsp": (
                partition.get("gsp")
                or partition.get(
                    "gsp_shard"
                )
                or partition.get(
                    "start_row"
                )
            ),

            "xor": (
                partition.get("xor")
                or partition.get(
                    "xor_shard"
                )
            ),

            "quorum": partition.get(
                "quorum"
            ),
        }

        return deterministic_vector(
            routing,
            size=self.vector_size,
        )

    # ========================================================================
    # FULL PARTITION VECTOR
    # ========================================================================

    def partition_vector(
        self,
        partition: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        """
        Build the complete GridCV representation.

        This is the primary bridge from memory_partition.py
        into downstream brain layers.
        """

        validation = self.validate_partition(
            partition
        )

        vectors = {

            "routing": self.routing_vector(
                partition
            ),

            "domain": self.domain_vector(
                partition.get(
                    "domain"
                )
            ),

          