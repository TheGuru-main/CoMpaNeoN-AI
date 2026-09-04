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
            "domain"
        )

        or result.get(
            "primary_domain"
        )

        or "general"

    )

    intent = (

        result.get(
            "intent"
        )

        or result.get(
            "primary_intent"
        )

        or result.get(
            "intent_type"
        )

        or "general"

    )

    result.setdefault(
        "domain",
        domain,
    )

    result.setdefault(
        "intent",
        intent,
    )

    return result


# ============================================================================
# DIRECTIVE SIGNAL
# ============================================================================

def directive_signal(
    text: str,
) -> Dict[
    str,
    Any,
]:

    directive = "general"

    if callable(
        detect_directive
    ):

        try:

            directive = (
                detect_directive(
                    text
                )
            )

        except Exception:

            directive = "general"

    lowered = str(
        text
    ).lower()

    matched = [

        phrase

        for phrase
        in DIRECTIVES

        if str(
            phrase
        ).lower()
        in lowered

    ]

    return {

        "directive": directive,

        "matched": matched,

        "count": len(
            matched
        ),

    }


# ============================================================================
# SYMBOL SIGNAL
# ============================================================================

def symbol_signal(
    text: str,
    domain: str = "general",
) -> Dict[
    str,
    Any,
]:

    symbols: List[
        Any
    ] = []

    if callable(
        recognize_symbols
    ):

        try:

            symbols = list(
                recognize_symbols(
                    text,
                    domain,
                )
            )

        except Exception:

            symbols = []

    return {

        "domain": domain,

        "symbols": symbols,

        "count": len(
            symbols
        ),

    }


# ============================================================================
# CODE LANGUAGE SIGNAL
# ============================================================================

def code_language_signal(
    text: str,
) -> Dict[
    str,
    Any,
]:

    lowered = str(
        text
    ).lower()

    languages: List[
        str
    ] = []

    terms: Dict[
        str,
        Any,
    ] = {}

    if callable(
        get_language_list
    ):

        try:

            for language in (
                get_language_list()
            ):

                if (
                    str(
                        language
                    ).lower()
                    in lowered
                ):

                    languages.append(
                        str(
                            language
                        )
                    )

        except Exception:

            pass

    if callable(
        get_code_terms
    ):

        try:

            for term, meaning in (
                get_code_terms().items()
            ):

                if (
                    str(
                        term
                    ).lower()
                    in lowered
                ):

                    terms[
                        str(term)
                    ] = meaning

        except Exception:

            pass

    return {

        "languages": sorted(
            set(
                languages
            )
        ),

        "terms": terms,

        "language_count": len(
            set(
                languages
            )
        ),

        "term_count": len(
            terms
        ),

    }


# ============================================================================
# EXTERNAL SIGNAL
# ============================================================================

def external_signal(
    source: Optional[
        str
    ] = None,
) -> Dict[
    str,
    Any,
]:
    """
    Calls the external architecture as an availability signal.

    This does NOT fetch remote information.

    WebCrawler remains responsible for acquisition.
    external.py remains responsible for source adapters.
    """

    if external_module is None:

        return {

            "available": False,

            "source": source,

            "adapter": None,

        }

    requested = (

        str(
            source
        ).strip().lower()

        if source

        else ""

    )

    adapters = sorted(

        name

        for name
        in dir(
            external_module
        )

        if (
            name.startswith(
                "fetch_"
            )

            or name.startswith(
                "search_"
            )
        )

    )

    adapter = None

    if requested:

        candidates = (

            f"fetch_{requested}",

            f"search_{requested}",

            requested,

        )

        for candidate in candidates:

            if callable(

                getattr(
                    external_module,
                    candidate,
                    None,
                )

            ):

                adapter = candidate

                break

    return {

        "available": True,

        "source": (
            requested
            or None
        ),

        "adapter": adapter,

        "adapter_count": len(
            adapters
        ),

        "adapters": adapters,

    }


# ============================================================================
# DATA MIXER SIGNAL
# ============================================================================

def data_mixer_signal(
    text: str,
    lang: Optional[
        str
    ] = None,
    source: str = "matrix_maths",
    mixer: Optional[
        Any
    ] = None,
) -> Dict[
    str,
    Any,
]:

    active = mixer

    if (
        active is None
        and DataMixer is not None
    ):

        try:

            active = DataMixer()

        except Exception:

            active = None

    if active is None:

        return {

            "available": False,

            "record": {},

        }

    methods = (

        "mix",

        "mix_record",

        "prepare",

        "prepare_record",

        "process",

    )

    for method_name in methods:

        method = getattr(
            active,
            method_name,
            None,
        )

        if not callable(
            method
        ):

            continue

        try:

            value = method(
                text,
                lang=lang,
                source=source,
            )

        except TypeError:

            try:

                value = method(
                    text
                )

            except Exception:

                continue

        except Exception:

            continue

        return {

            "available": True,

            "method": method_name,

            "record": (

                value

                if isinstance(
                    value,
                    dict,
                )

                else {
                    "value": value
                }

            ),

        }

    return {

        "available": True,

        "method": None,

        "record": {},

    }


# ============================================================================
# GRID CV SIGNAL
# ============================================================================

def grid_cv_signal(
    partition: Optional[
        Any
    ] = None,
    grid_cv: Optional[
        Any
    ] = None,
    vector_size: int = (
        DEFAULT_SIGNAL_VECTOR_SIZE
    ),
) -> Dict[
    str,
    Any,
]:

    active = grid_cv

    if (
        active is None
        and GridCV is not None
    ):

        try:

            active = GridCV()

        except Exception:

            active = None

    if partition is None:

        return {

            "available": (
                active is not None
            ),

            "value": {},

            "vector": (
                deterministic_vector(
                    {
                        "partition": None
                    },
                    vector_size,
                )
            ),

        }

    if active is not None:

        methods = (

            "analyze",

            "validate",

            "represent",

            "vectorize",

            "partition_vector",

            "project_context_vector",

        )

        for method_name in methods:

            method = getattr(
                active,
                method_name,
                None,
            )

            if not callable(
                method
            ):

                continue

            try:

                value = method(
                    partition
                )

            except Exception:

                continue

            payload = (

                value

                if isinstance(
                    value,
                    dict,
                )

                else {
                    "value": value
                }

            )

            return {

                "available": True,

                "method": method_name,

                "value": payload,

                "vector": (
                    deterministic_vector(
                        payload,
                        vector_size,
                    )
                ),

            }

    payload = (

        partition

        if isinstance(
            partition,
            dict,
        )

        else {
            "value": partition
        }

    )

    return {

        "available": (
            active is not None
        ),

        "value": payload,

        "vector": (
            deterministic_vector(
                payload,
                vector_size,
            )
        ),

    }


# ============================================================================
# CATEGORICAL MATRIX
# ============================================================================

def categorical_matrix(
    values: Sequence[
        Any
    ],
    vector_size: int = (
        DEFAULT_SIGNAL_VECTOR_SIZE
    ),
) -> torch.Tensor:

    rows = [

        deterministic_vector(
            value,
            vector_size,
        )

        for value
        in values

    ]

    if not rows:

        return torch.zeros(

            (
                0,
                int(
                    vector_size
                ),
            ),

            dtype=torch.float32,

        )

    return torch.stack(
        rows
    )


# ============================================================================
# SIGNAL SIMILARITY MATRIX
# ============================================================================

def signal_similarity_matrix(
    signals: Mapping[
        str,
        Any,
    ],
    vector_size: int = (
        DEFAULT_SIGNAL_VECTOR_SIZE
    ),
) -> Tuple[
    List[str],
    torch.Tensor,
]:

    names = list(
        signals.keys()
    )

    if not names:

        return (

            [],

            torch.zeros(
                (
                    0,
                    0,
                ),
                dtype=torch.float32,
            ),

        )

    vectors = {

        name: deterministic_vector(
            signals[
                name
            ],
            vector_size,
        )

        for name
        in names

    }

    matrix = torch.zeros(

        (
            len(
                names
            ),
            len(
                names
            ),
        ),

        dtype=torch.float32,

    )

    for i, left_name in enumerate(
        names
    ):

        for j, right_name in enumerate(
            names
        ):

            matrix[
                i,
                j,
            ] = cosine_similarity(

                vectors[
                    left_name
                ],

                vectors[
                    right_name
                ],

            )

    return (

        names,

        matrix,

    )


# ============================================================================
# RELATIONSHIP SIGNAL MATRIX
# ============================================================================

def relationship_signal_matrix(
    relationships: Sequence[
        Any
    ],
    vector_size: int = (
        DEFAULT_SIGNAL_VECTOR_SIZE
    ),
) -> Dict[
    str,
    Any,
]:

    labels = [

        str(
            item
        )

        for item
        in relationships

    ]

    matrix = categorical_matrix(

        labels,

        vector_size,

    )

    if matrix.shape[0] == 0:

        similarity = torch.zeros(

            (
                0,
                0,
            ),

            dtype=torch.float32,

        )

    else:

        similarity = torch.matmul(

            matrix,

            matrix.T,

        )

    return {

        "relationships": labels,

        "matrix": matrix,

        "similarity": (

            normalize_matrix(
                similarity
            )

            if similarity.numel()

            else similarity

        ),

    }


# ============================================================================
# DOMAIN SCORE
# ============================================================================

def domain_score(
    domain: str,
    candidate_domain: str,
) -> float:

    if (
        not domain
        or not candidate_domain
    ):

        return 0.0

    return (

        1.0

        if (
            str(
                domain
            ).lower()

            == str(
                candidate_domain
            ).lower()
        )

        else 0.0

    )


# ============================================================================
# INTENT SCORE
# ============================================================================

def intent_score(
    intent: str,
    candidate_intent: str,
) -> float:

    if (
        not intent
        or not candidate_intent
    ):

        return 0.0

    return (

        1.0

        if (
            str(
                intent
            ).lower()

            == str(
                candidate_intent
            ).lower()
        )

        else 0.0

    )


# ============================================================================
# SYMBOL SCORE
# ============================================================================

def symbol_score(
    left: Sequence[
        Any
    ],
    right: Sequence[
        Any
    ],
) -> float:

    left_set = {

        str(

            item[0]

            if (
                isinstance(
                    item,
                    (
                        tuple,
                        list,
                    ),
                )
                and item
            )

            else item

        ).lower()

        for item
        in left

    }

    right_set = {

        str(

            item[0]

            if (
                isinstance(
                    item,
                    (
                        tuple,
                        list,
                    ),
                )
                and item
            )

            else item

        ).lower()

        for item
        in right

    }

    if (
        not left_set
        or not right_set
    ):

        return 0.0

    return (

        len(
            left_set
            & right_set
        )

        / len(
            left_set
            | right_set
        )

    )


def relationship_score(
    relationships_a: Sequence[
        Any
    ],
    relationships_b: Sequence[
        Any
    ],
) -> float:

    return symbol_score(

        relationships_a,

        relationships_b,

    )


# ============================================================================
# SIGNAL WEIGHTED SCORE
# ============================================================================

def signal_weighted_score(
    signal_scores: Mapping[
        str,
        float,
    ],
    weights: Optional[
        Mapping[
            str,
            float,
        ]
    ] = None,
) -> float:

    if not signal_scores:

        return 0.0

    weights = (
        weights
        or {}
    )

    values: List[
        float
    ] = []

    used_weights: List[
        float
    ] = []

    for name, score in (
        signal_scores.items()
    ):

        values.append(
            float(
                score
            )
        )

        used_weights.append(

            float(

                weights.get(
                    name,
                    1.0,
                )

            )

        )

    return (
        weighted_relationship_score(
            values,
            used_weights,
        )
    )


# ============================================================================
# COMPLETE MATRIX ANALYSIS
# ============================================================================

def analyze_matrix_signals(
    text: str,
    lang: Optional[
        str
    ] = None,
    alphabet: Optional[
        str
    ] = None,
    alphabet_relationships: Optional[
        Mapping[
            int,
            Sequence[str],
        ]
    ] = None,
    relationships: Optional[
        Sequence[
            Any
        ]
    ] = None,
    partition: Optional[
        Any
    ] = None,
    external_source: Optional[
        str
    ] = None,
    data_mixer: Optional[
        Any
    ] = None,
    grid_cv: Optional[
        Any
    ] = None,
    vector_size: int = (
        DEFAULT_SIGNAL_VECTOR_SIZE
    ),
) -> Dict[
    str,
    Any,
]:
    """
    Unified Matrix Maths entry.

    This function connects the already-established architecture without
    taking ownership from the individual modules.
    """

    text = str(
        text
        or ""
    )

    intent_data = (
        intent_signal(
            text
        )
    )

    domain = str(

        intent_data.get(
            "domain"
        )

        or "general"

    )

    intent = str(

        intent_data.get(
            "intent"
        )

        or "general"

    )

    directive_data = (
        directive_signal(
            text
        )
    )

    symbols_data = (
        symbol_signal(
            text,
            domain,
        )
    )

    code_data = (
        code_language_signal(
            text
        )
    )

    external_data = (
        external_signal(
            external_source
        )
    )

    mixed_data = (
        data_mixer_signal(
            text,
            lang=lang,
            mixer=data_mixer,
        )
    )

    cv_data = (
        grid_cv_signal(
            partition,
            grid_cv,
            vector_size,
        )
    )

    relationship_data = (
        relationship_signal_matrix(

            relationships
            or (),

            vector_size,

        )
    )

    alphabet_data: Dict[
        str,
        Any,
    ] = {

        "available": False,

        "alphabet": (
            alpha