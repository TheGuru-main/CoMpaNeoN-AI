"""
CoMpaNeoN Matrix Mathematics
============================

Unified mathematical signal layer.

Matrix Maths receives already-established signals from:

    intent_analyzer.py
    symbols.py
    directives.py
    external.py
    code_languages.py
    data_mixer.py
    grid_cv.py

and converts those signals into deterministic mathematical
representations for:

    relation_and_alphabet_matrix.py
    follow_up.py
    summary.py
    linguistic layers
    semantic layers
    WordUnderstanding
    brain constituents

ARCHITECTURE
------------

intent_analyzer.py
symbols.py
directives.py
external.py
code_languages.py
data_mixer.py
grid_cv.py
        │
        ▼
matrix_maths.py
        │
        ├── alphabet mathematics
        ├── relationship mathematics
        ├── domain signals
        ├── intent signals
        ├── directive signals
        ├── symbol signals
        ├── code-language signals
        ├── external-source signals
        ├── data-mixer signals
        └── GridCV signals
                │
                ▼
relation_and_alphabet_matrix.py
                │
                ▼
follow_up.py / summary.py
                │
                ▼
linguistic + semantic layers
                │
                ▼
understanding + brain constituents

AUTHORITY
---------

tokenizer.py
    owns linguistic tokenization and language mapping.

intent_analyzer.py
    owns domain/entity/intent analysis.

symbols.py
    owns symbol and abbreviation recognition.

directives.py
    owns directive recognition.

code_languages.py
    owns programming language knowledge.

data_mixer.py
    owns data preparation.

external.py
    owns external source adapters.

grid_cv.py
    owns partition validation and partition-compatible vectors.

relation_and_alphabet_matrix.py
    owns the multilingual alphabet/relationship substrate.

This module does NOT:

    - tokenize text
    - detect crawler paths
    - perform GSP placement
    - traverse MemoryGrid
    - index MemoryGrid
    - rank retrieval
    - generate prompts
    - execute tools
    - generate AI responses

Matrix Maths is the mathematical signal layer between the established
analysis/partition layers and the relationship/alphabet matrix.
"""

from __future__ import annotations

import hashlib
import json
import os

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch


# ============================================================================
# ARCHITECTURE IMPORTS
# ============================================================================

try:

    from intent_analyzer import (
        analyze_intent,
    )

except Exception:

    analyze_intent = None


try:

    from symbols import (
        recognize_symbols,
    )

except Exception:

    recognize_symbols = None


try:

    from directives import (
        DIRECTIVES,
        detect_directive,
    )

except Exception:

    DIRECTIVES = {}
    detect_directive = None


try:

    from code_languages import (
        get_code_terms,
        get_language_list,
    )

except Exception:

    get_code_terms = None
    get_language_list = None


try:

    from data_mixer import (
        DataMixer,
    )

except Exception:

    DataMixer = None


try:

    import external as external_module

except Exception:

    external_module = None


try:

    from grid_cv import (
        GridCV,
    )

except Exception:

    GridCV = None


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_WEIGHT = 1.0

MIN_WEIGHT = 0.0

MAX_WEIGHT = 1.0

DEFAULT_EXAMPLE_COUNT = 2000

DEFAULT_SIGNAL_VECTOR_SIZE = 64


SIGNAL_NAMES = (

    "alphabet",

    "relationship",

    "domain",

    "intent",

    "directive",

    "symbol",

    "code_language",

    "external",

    "data_mixer",

    "grid_cv",

)


# ============================================================================
# BASIC NUMERICAL UTILITIES
# ============================================================================

def clamp(
    value: float,
    minimum: float = MIN_WEIGHT,
    maximum: float = MAX_WEIGHT,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def normalize_weight(
    value: float,
    maximum: float = 1.0,
) -> float:

    if maximum <= 0:
        return 0.0

    return clamp(
        float(value)
        / float(maximum)
    )


def _as_tensor(
    value: Any,
    ndim: Optional[int] = None,
) -> torch.Tensor:

    if isinstance(
        value,
        torch.Tensor,
    ):

        tensor = value.float()

    else:

        tensor = torch.tensor(
            value,
            dtype=torch.float32,
        )

    if (
        ndim is not None
        and tensor.ndim != ndim
    ):

        raise ValueError(
            f"Expected {ndim} dimensions, "
            f"received {tensor.ndim}."
        )

    return tensor


def _stable_json(
    value: Any,
) -> str:

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

    return hashlib.sha256(
        _stable_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================================
# DETERMINISTIC SIGNAL VECTOR
# ============================================================================

def deterministic_vector(
    value: Any,
    size: int = DEFAULT_SIGNAL_VECTOR_SIZE,
) -> torch.Tensor:
    """
    Convert a deterministic architecture signal into a stable vector.

    This is not a learned embedding.

    It gives:

        domain
        intent
        symbols
        relationships
        partition state

    stable mathematical representations.
    """

    size = max(
        1,
        int(size),
    )

    seed = _stable_hash(
        value
    )

    values: List[
        float
    ] = []

    for index in range(
        size
    ):

        digest = hashlib.sha256(
            f"{seed}:{index}".encode(
                "utf-8"
            )
        ).digest()

        integer = int.from_bytes(
            digest[:8],
            "big",
            signed=False,
        )

        value_float = (
            integer
            / float(
                (1 << 64) - 1
            )
        )

        values.append(
            value_float
            * 2.0
            - 1.0
        )

    vector = torch.tensor(
        values,
        dtype=torch.float32,
    )

    norm = torch.norm(
        vector
    )

    if norm.item() > 0:

        vector = (
            vector
            / norm
        )

    return vector


def cosine_similarity(
    vector_a: Any,
    vector_b: Any,
) -> float:

    a = _as_tensor(
        vector_a
    ).flatten()

    b = _as_tensor(
        vector_b
    ).flatten()

    if (
        a.numel()
        != b.numel()
    ):

        size = min(
            a.numel(),
            b.numel(),
        )

        if size <= 0:
            return 0.0

        a = a[:size]

        b = b[:size]

    a_norm = torch.norm(
        a
    )

    b_norm = torch.norm(
        b
    )

    if (
        a_norm.item() == 0
        or b_norm.item() == 0
    ):

        return 0.0

    return float(
        torch.clamp(
            torch.dot(
                a,
                b,
            )
            / (
                a_norm
                * b_norm
            ),
            -1.0,
            1.0,
        ).item()
    )


# ============================================================================
# RELATIONSHIP WEIGHT
# ============================================================================

def relationship_weight(
    class_id: int,
    pair_position: int = 0,
    class_count: int = 25,
) -> float:

    class_id = max(
        1,
        min(
            int(class_id),
            int(class_count),
        ),
    )

    pair_position = max(
        0,
        int(pair_position),
    )

    class_component = (
        class_count
        - class_id
        + 1
    ) / class_count

    pair_component = (
        1.0
        / (
            pair_position
            + 2.0
        )
    )

    return clamp(
        class_component
        * 0.85
        + pair_component
        * 0.15
    )


# ============================================================================
# ALPHABET MATRIX
# ============================================================================

def build_alphabet_matrix(
    alphabet: str,
    relationship_pairs: Optional[
        Mapping[
            int,
            Sequence[str],
        ]
    ] = None,
) -> torch.Tensor:

    alphabet = "".join(
        dict.fromkeys(
            str(
                alphabet
            ).strip()
        )
    )

    if not alphabet:

        raise ValueError(
            "Alphabet cannot be empty."
        )

    size = len(
        alphabet
    )

    matrix = torch.zeros(
        (
            size,
            size,
        ),
        dtype=torch.float32,
    )

    indices = {

        letter.lower(): position

        for position, letter
        in enumerate(
            alphabet
        )

    }

    for position in range(
        size
    ):

        matrix[
            position,
            position,
        ] = DEFAULT_WEIGHT

    if not relationship_pairs:

        return matrix

    for class_id, pairs in (
        relationship_pairs.items()
    ):

        for pair_position, pair in enumerate(
            pairs
        ):

            pair = str(
                pair
            ).strip()

            if len(pair) != 2:
                continue

            first = pair[
                0
            ].lower()

            second = pair[
                1
            ].lower()

            if (
                first
                not in indices
                or second
                not in indices
            ):

                continue

            weight = relationship_weight(
                class_id=int(
                    class_id
                ),
                pair_position=pair_position,
            )

            i = indices[
                first
            ]

            j = indices[
                second
            ]

            matrix[
                i,
                j,
            ] = max(
                matrix[
                    i,
                    j,
                ],
                torch.tensor(
                    weight
                ),
            )

            matrix[
                j,
                i,
            ] = max(
                matrix[
                    j,
                    i,
                ],
                torch.tensor(
                    weight
                ),
            )

    return matrix


# ============================================================================
# MATRIX NORMALIZATION
# ============================================================================

def normalize_matrix(
    matrix: Any,
) -> torch.Tensor:

    matrix = _as_tensor(
        matrix
    )

    maximum = torch.max(
        torch.abs(
            matrix
        )
    )

    if maximum.item() == 0:

        return matrix.clone()

    return (
        matrix
        / maximum
    )


def row_normalize(
    matrix: Any,
) -> torch.Tensor:

    matrix = _as_tensor(
        matrix,
        ndim=2,
    )

    denominator = matrix.sum(
        dim=1,
        keepdim=True,
    )

    denominator = torch.where(
        denominator == 0,
        torch.ones_like(
            denominator
        ),
        denominator,
    )

    return (
        matrix
        / denominator
    )


# ============================================================================
# MATRIX MULTIPLICATION
# ============================================================================

def matrix_multiply(
    left: Any,
    right: Any,
) -> torch.Tensor:

    left = _as_tensor(
        left,
        ndim=2,
    )

    right = _as_tensor(
        right,
        ndim=2,
    )

    if (
        left.shape[1]
        != right.shape[0]
    ):

        raise ValueError(
            "Matrix dimensions are incompatible: "
            f"{tuple(left.shape)} "
            f"@ {tuple(right.shape)}"
        )

    return torch.matmul(
        left,
        right,
    )


def matrix_vector_multiply(
    matrix: Any,
    vector: Any,
) -> torch.Tensor:

    matrix = _as_tensor(
        matrix,
        ndim=2,
    )

    vector = _as_tensor(
        vector,
        ndim=1,
    )

    if (
        matrix.shape[1]
        != vector.shape[0]
    ):

        raise ValueError(
            "Matrix and vector dimensions "
            "are incompatible."
        )

    return torch.matmul(
        matrix,
        vector,
    )


# ============================================================================
# RELATIONSHIP PROPAGATION
# ============================================================================

def propagate_relationships(
    matrix: Any,
    vector: Any,
    steps: int = 1,
    normalize: bool = True,
) -> torch.Tensor:

    result = _as_tensor(
        vector,
        ndim=1,
    )

    for _ in range(
        max(
            0,
            int(steps),
        )
    ):

        result = (
            matrix_vector_multiply(
                matrix,
                result,
            )
        )

        if normalize:

            maximum = torch.max(
                torch.abs(
                    result
                )
            )

            if maximum.item() > 0:

                result = (
                    result
                    / maximum
                )

    return result


# ============================================================================
# RELATIONSHIP SIMILARITY
# ============================================================================

def relationship_similarity(
    vector_a: Any,
    vector_b: Any,
) -> float:

    return (
        cosine_similarity(
            vector_a,
            vector_b,
        )
        * 100.0
    )


# ============================================================================
# WEIGHTED RELATIONSHIP SCORE
# ============================================================================

def weighted_relationship_score(
    values: Iterable[
        float
    ],
    weights: Iterable[
        float
    ],
) -> float:

    values = list(
        values
    )

    weights = list(
        weights
    )

    if (
        not values
        or not weights
    ):

        return 0.0

    if (
        len(values)
        != len(weights)
    ):

        raise ValueError(
            "values and weights "
            "must have equal lengths."
        )

    denominator = sum(
        float(weight)
        for weight
        in weights
    )

    if denominator == 0:

        return 0.0

    numerator = sum(

        float(value)
        * float(weight)

        for value, weight
        in zip(
            values,
            weights,
        )

    )

    return (
        numerator
        / denominator
    )


# ============================================================================
# ALPHABET VECTORS
# ============================================================================

def alphabet_one_hot(
    letter: str,
    alphabet: str,
) -> torch.Tensor:

    alphabet = str(
        alphabet
    )

    vector = torch.zeros(
        len(
            alphabet
        ),
        dtype=torch.float32,
    )

    try:

        index = (
            alphabet.lower().index(
                str(
                    letter
                ).strip().lower()
            )
        )

    except ValueError:

        return vector

    vector[
        index
    ] = 1.0

    return vector


def letter_relationship_vector(
    letter: str,
    alphabet: str,
    matrix: Any,
) -> torch.Tensor:

    return (
        matrix_vector_multiply(
            matrix,
            alphabet_one_hot(
                letter,
                alphabet,
            ),
        )
    )


# ============================================================================
# WORD MATRIX SCORE
# ============================================================================

def word_matrix_score(
    word: str,
    alphabet: str,
    matrix: Any,
) -> float:

    word = str(
        word
    ).strip().lower()

    alphabet = str(
        alphabet
    ).lower()

    matrix = _as_tensor(
        matrix,
        ndim=2,
    )

    if len(word) < 2:

        return 0.0

    indices = {

        letter: index

        for index, letter
        in enumerate(
            alphabet
        )

    }

    scores: List[
        float
    ] = []

    for index in range(
        len(word)
        - 1
    ):

        first = word[
            index
        ]

        second = word[
            index + 1
        ]

        if (
            first
            not in indices
            or second
            not in indices
        ):

            continue

        scores.append(
            float(
                matrix[
                    indices[first],
                    indices[second],
                ].item()
            )
        )

    if not scores:

        return 0.0

    return (
        sum(scores)
        / len(scores)
    ) * 100.0


def matrix_features(
    word: str,
    alphabet: str,
    matrix: Any,
) -> Dict[
    str,
    Any,
]:

    word = str(
        word
    ).strip().lower()

    matrix = _as_tensor(
        matrix,
        ndim=2,
    )

    pairs: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for index in range(
        max(
            0,
            len(word)
            - 1,
        )
    ):

        pair = word[
            index:index + 2
        ]

        try:

            first_index = (
                alphabet.lower().index(
                    pair[0]
                )
            )

            second_index = (
                alphabet.lower().index(
                    pair[1]
                )
            )

            weight = float(
                matrix[
                    first_index,
                    second_index,
                ].item()
            )

        except (
            ValueError,
            IndexError,
        ):

            weight = 0.0

        pairs.append({

            "pair": pair,

            "weight": weight,

        })

    return {

        "word": word,

        "matrix_score": (
            word_matrix_score(
                word,
                alphabet,
                matrix,
            )
        ),

        "pair_count": len(
            pairs
        ),

        "pairs": pairs,

    }


# ============================================================================
# INTENT ANALYZER SIGNAL
# ============================================================================

def intent_signal(
    text: str,
) -> Dict[
    str,
    Any,
]:

    result: Dict[
        str,
        Any,
    ] = {}

    if callable(
        analyze_intent
    ):

        try:

            analyzed = (
                analyze_intent(
                    text
                )
            )

            if isinstance(
                analyzed,
                dict,
            ):

                result = dict(
                    analyzed
                )

            else:

                result = {
                    "value": analyzed
                }

        except Exception as exc:

            result = {
                "error": str(exc)
            }

    domain = (

        result.get(
            "dom