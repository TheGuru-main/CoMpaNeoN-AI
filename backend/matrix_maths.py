"""
CoMpaNeoN Matrix Mathematics
============================

Mathematical substrate for alphabet_matrix.py.

Responsibilities:
- Convert alphabet relationships into numerical matrices.
- Generate deterministic relationship weights.
- Perform matrix/vector multiplication.
- Perform matrix/matrix multiplication.
- Normalize relationship matrices.
- Calculate weighted relationship propagation.
- Generate 2,000 deterministic mathematical examples.
- Validate generated examples.
- Provide serializable training/deployment metadata.

Architecture:

    tokenizer.py
          |
          v
    alphabet_matrix.py
          |
          v
    matrix_maths.py
          |
          +---------> ranking.py
          |
          +---------> word_chain.py
                         |
                         v
                 word_understanding.py

Important:
- This module does NOT generate AI responses.
- This module does NOT replace tokenizer.py.
- This module does NOT replace alphabet_matrix.py.
- This module does NOT replace ranking.py.
- This module does NOT access MemoryGrid.
- This module does NOT modify ai_model.py.

The numerical matrices produced here may later become part of the
neural computation layer, but that integration is deliberately deferred.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

DEFAULT_EXAMPLE_COUNT = 2000


# ---------------------------------------------------------------------------
# BASIC NUMERICAL UTILITIES
# ---------------------------------------------------------------------------

def clamp(
    value: float,
    minimum: float = MIN_WEIGHT,
    maximum: float = MAX_WEIGHT,
) -> float:
    """
    Clamp a numerical value into a deterministic range.
    """

    return max(
        minimum,
        min(maximum, float(value)),
    )


def normalize_weight(
    value: float,
    maximum: float = 1.0,
) -> float:
    """
    Normalize a value into the range 0.0 - 1.0.
    """

    if maximum <= 0:
        return 0.0

    return clamp(
        float(value) / float(maximum)
    )


# ---------------------------------------------------------------------------
# RELATIONSHIP WEIGHT
# ---------------------------------------------------------------------------

def relationship_weight(
    class_id: int,
    pair_position: int = 0,
    class_count: int = 25,
) -> float:
    """
    Generate a deterministic base weight for a relationship class.

    Higher relationship classes receive a stable mathematical weighting
    rather than a random value.

    pair_position allows individual pairs inside the same class to receive
    a deterministic secondary weighting.
    """

    class_id = max(
        1,
        min(int(class_id), class_count),
    )

    pair_position = max(
        0,
        int(pair_position),
    )

    # Primary class contribution.
    class_component = (
        class_count - class_id + 1
    ) / class_count

    # Small deterministic pair contribution.
    pair_component = 1.0 / (
        pair_position + 2.0
    )

    weight = (
        class_component * 0.85
        + pair_component * 0.15
    )

    return clamp(weight)


# ---------------------------------------------------------------------------
# BUILD NUMERICAL ALPHABET MATRIX
# ---------------------------------------------------------------------------

def build_alphabet_matrix(
    alphabet: str,
    relationship_pairs: Optional[
        Dict[int, Sequence[str]]
    ] = None,
) -> torch.Tensor:
    """
    Build a square numerical alphabet relationship matrix.

    Each character occupies one row and one column.

    If a relationship pair exists:

        A -> B

    then:

        matrix[A, B] = weight
        matrix[B, A] = weight

    The matrix is symmetric by design because alphabet relationship
    membership is treated as bidirectional at this mathematical layer.

    Diagonal entries represent self-relationship and are initialized
    to 1.0.
    """

    alphabet = "".join(
        dict.fromkeys(
            str(alphabet).strip()
        )
    )

    if not alphabet:
        raise ValueError(
            "Alphabet cannot be empty."
        )

    size = len(alphabet)

    matrix = torch.zeros(
        (size, size),
        dtype=torch.float32,
    )

    index = {
        letter: i
        for i, letter in enumerate(alphabet)
    }

    # Self relationships.
    for i in range(size):
        matrix[i, i] = DEFAULT_WEIGHT

    if not relationship_pairs:
        return matrix

    for class_id, pairs in relationship_pairs.items():

        for position, pair in enumerate(pairs):

            pair = str(pair).strip().upper()

            if len(pair) != 2:
                continue

            first = pair[0].lower()
            second = pair[1].lower()

            if (
                first not in index
                or second not in index
            ):
                continue

            weight = relationship_weight(
                class_id=class_id,
                pair_position=position,
            )

            i = index[first]
            j = index[second]

            matrix[i, j] = max(
                matrix[i, j],
                weight,
            )

            matrix[j, i] = max(
                matrix[j, i],
                weight,
            )

    return matrix


# ---------------------------------------------------------------------------
# MATRIX NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_matrix(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize a matrix by its largest absolute value.
    """

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(
            matrix,
            dtype=torch.float32,
        )

    maximum = torch.max(
        torch.abs(matrix)
    )

    if maximum.item() == 0:
        return matrix.clone()

    return matrix / maximum


# ---------------------------------------------------------------------------
# ROW NORMALIZATION
# ---------------------------------------------------------------------------

def row_normalize(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize every row independently.

    This is useful when interpreting each alphabet character as a
    distribution over related characters.
    """

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(
            matrix,
            dtype=torch.float32,
        )

    denominator = matrix.sum(
        dim=1,
        keepdim=True,
    )

    denominator = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        denominator,
    )

    return matrix / denominator


# ---------------------------------------------------------------------------
# MATRIX MULTIPLICATION
# ---------------------------------------------------------------------------

def matrix_multiply(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic matrix multiplication.

        C = A @ B

    This is the fundamental operation that will later allow the
    alphabet relationship substrate to participate in neural computation.
    """

    if not isinstance(left, torch.Tensor):
        left = torch.tensor(
            left,
            dtype=torch.float32,
        )

    if not isinstance(right, torch.Tensor):
        right = torch.tensor(
            right,
            dtype=torch.float32,
        )

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            "matrix_multiply requires two 2-dimensional matrices."
        )

    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "Matrix dimensions are incompatible: "
            f"{tuple(left.shape)} @ {tuple(right.shape)}"
        )

    return torch.matmul(
        left,
        right,
    )


# ---------------------------------------------------------------------------
# MATRIX-VECTOR MULTIPLICATION
# ---------------------------------------------------------------------------

def matrix_vector_multiply(
    matrix: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply an alphabet relationship matrix by a vector.

        y = A @ x
    """

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(
            matrix,
            dtype=torch.float32,
        )

    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(
            vector,
            dtype=torch.float32,
        )

    if matrix.ndim != 2:
        raise ValueError(
            "Matrix must be 2-dimensional."
        )

    if vector.ndim != 1:
        raise ValueError(
            "Vector must be 1-dimensional."
        )

    if matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            "Matrix and vector dimensions are incompatible."
        )

    return torch.matmul(
        matrix,
        vector,
    )


# ---------------------------------------------------------------------------
# RELATIONSHIP PROPAGATION
# ---------------------------------------------------------------------------

def propagate_relationships(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    steps: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Propagate relationship activation through the alphabet matrix.

    Example:

        x
        |
        v
        A @ x
        |
        v
        A @ (A @ x)

    Each additional step allows relationships to travel farther through
    the matrix.
    """

    if steps < 1:
        return vector

    result = vector

    for _ in range(int(steps)):
        result = matrix_vector_multiply(
            matrix,
            result,
        )

        if normalize:
            maximum = torch.max(
                torch.abs(result)
            )

            if maximum.item() > 0:
                result = result / maximum

    return result


# ---------------------------------------------------------------------------
# RELATIONSHIP SIMILARITY
# ---------------------------------------------------------------------------

def relationship_similarity(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
) -> float:
    """
    Calculate cosine similarity between two relationship vectors.

    Returns:
        0.0 - 100.0
    """

    if not isinstance(vector_a, torch.Tensor):
        vector_a = torch.tensor(
            vector_a,
            dtype=torch.float32,
        )

    if not isinstance(vector_b, torch.Tensor):
        vector_b = torch.tensor(
            vector_b,
            dtype=torch.float32,
        )

    a_norm = torch.norm(vector_a)
    b_norm = torch.norm(vector_b)

    if (
        a_norm.item() == 0
        or b_norm.item() == 0
    ):
        return 0.0

    similarity = torch.dot(
        vector_a,
        vector_b,
    ) / (
        a_norm * b_norm
    )

    return float(
        torch.clamp(
            similarity,
            -1.0,
            1.0,
        ).item()
        * 100.0
    )


# ---------------------------------------------------------------------------
# WEIGHTED RELATIONSHIP SCORE
# ---------------------------------------------------------------------------

def weighted_relationship_score(
    values: Iterable[float],
    weights: Iterable[float],
) -> float:
    """
    Calculate a weighted arithmetic relationship score.
    """

    values = list(values)
    weights = list(weights)

    if not values or not weights:
        return 0.0

    if len(values) != len(weights):
        raise ValueError(
            "values and weights must have equal lengths."
        )

    numerator = sum(
        float(value) * float(weight)
        for value, weight
        in zip(values, weights)
    )

    denominator = sum(
        float(weight)
        for weight in weights
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ---------------------------------------------------------------------------
# ALPHABET ONE-HOT VECTOR
# ---------------------------------------------------------------------------

def alphabet_one_hot(
    letter: str,
    alphabet: str,
) -> torch.Tensor:
    """
    Convert a letter into a deterministic one-hot vector.
    """

    alphabet = str(alphabet)

    vector = torch.zeros(
        len(alphabet),
        dtype=torch.float32,
    )

    letter = str(letter).strip().lower()

    try:
        index = alphabet.lower().index(
            letter
        )
    except ValueError:
        return vector

    vector[index] = 1.0

    return vector


# ---------------------------------------------------------------------------
# LETTER RELATIONSHIP VECTOR
# ---------------------------------------------------------------------------

def letter_relationship_vector(
    letter: str,
    alphabet: str,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Project a letter into the relationship space defined by the matrix.
    """

    vector = alphabet_one_hot(
        letter,
        alphabet,
    )

    return matrix_vector_multiply(
        matrix,
        vector,
    )


# ---------------------------------------------------------------------------
# DETERMINISTIC EXAMPLE GENERATION
# ---------------------------------------------------------------------------

def generate_math_example(
    example_id: int,
    dimension: int,
) -> Dict[str, Any]:
    """
    Generate one deterministic matrix-mathematics example.

    No randomness is used.

    The generated example contains:

        - matrix dimensions
        - deterministic scalar weights
        - vector values
        - expected matrix multiplication result
        - normalized result
        - relationship score
    """

    if dimension < 2:
        raise ValueError(
            "Matrix dimension must be >= 2."
        )

    example_id = int(example_id)

    # Deterministic pseudo-pattern.
    scale = (
        (example_id % 17) + 1
    ) / 17.0

    matrix_values = []

    for row in range(dimension):

        current_row = []

        for col in range(dimension):

            base = (
                (
                    (example_id + 1)
                    * (row + 1)
                    * (col + 1)
                )
                % 19
            ) / 19.0

            diagonal_boost = (
                0.25
                if row == col
                else 0.0
            )

            value = clamp(
                base * scale
                + diagonal_boost
            )

            current_row.append(
                round(value, 6)
            )

        matrix_values.append(
            current_row
        )

    vector = [
        round(
            (
                (
                    example_id
                    + index
                    + 1
                )
                % 11
            ) / 10.0,
            6,
        )
        for index in range(dimension)
    ]

    matrix = torch.tensor(
        matrix_values,
        dtype=torch.float32,
    )

    vector_tensor = torch.tensor(
        vector,
        dtype=torch.float32,
    )

    result = matrix_vector_multiply(
        matrix,
        vector_tensor,
    )

    normalized = normalize_matrix(
        result.reshape(1, -1)
    ).flatten()

    weight = relationship_weight(
        class_id=((example_id - 1) % 25) + 1,
        pair_position=example_id % 13,
    )

    return {
        "id": example_id,
        "operation": "matrix_vector_multiplication",
        "dimension": dimension,
        "matrix": matrix_values,
        "vector": vector,
        "expected": [
            round(float(value), 6)
            for value in result.tolist()
        ],
        "normalized_expected": [
            round(float(value), 6)
            for value in normalized.tolist()
        ],
        "relationship_class": (
            ((example_id - 1) % 25) + 1
        ),
        "weight": round(
            float(weight),
            6,
        ),
    }


# ---------------------------------------------------------------------------
# GENERATE 2000 EXAMPLES
# ---------------------------------------------------------------------------

def generate_training_examples(
    count: int = DEFAULT_EXAMPLE_COUNT,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic mathematical training examples.

    Default:
        2,000 examples.

    Dimensions cycle through several matrix sizes so the dataset exercises
    more than one mathematical shape.
    """

    count = int(count)

    if count <= 0:
        return []

    examples = []

    dimensions = (
        2,
        3,
        4,
        5,
        6,
        8,
    )

    for example_id in range(
        1,
        count + 1,
    ):

        dimension = dimensions[
            (example_id - 1)
            % len(dimensions)
        ]

        examples.append(
            generate_math_example(
                example_id,
                dimension,
            )
        )

    return examples


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_math_example(
    example: Dict[str, Any],
) -> bool:
    """
    Recalculate an example and verify its stored expected result.
    """

    matrix = torch.tensor(
        example["matrix"],
        dtype=torch.float32,
    )

    vector = torch.tensor(
        example["vector"],
        dtype=torch.float32,
    )

    expected = torch.tensor(
        example["expected"],
        dtype=torch.float32,
    )

    calculated = matrix_vector_multiply(
        matrix,
        vector,
    )

    return bool(
        torch.allclose(
            calculated,
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
    )


def validate_training_examples(
    examples: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Validate every generated mathematical example.
    """

    valid = 0
    invalid = []

    for example in examples:

        if validate_math_example(
            example
        ):
            valid += 1
        else:
            invalid.append(
                example.get("id")
            )

    return {
        "total": len(examples),
        "valid": valid,
        "invalid": len(invalid),
        "invalid_ids": invalid,
        "passed": len(invalid) == 0,
    }


# ---------------------------------------------------------------------------
# DATASET METADATA
# ---------------------------------------------------------------------------

def training_metadata(
    examples: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Return metadata for the deterministic matrix mathematics dataset.
    """

    dimensions = sorted(
        {
            int(example["dimension"])
            for example in examples
        }
    )

    classes = sorted(
        {
            int(example["relationship_class"])
            for example in examples
        }
    )

    validation = validate_training_examples(
        examples
    )

    return {
        "dataset": "CoMpaNeoN Alphabet Matrix Mathematics",
        "version": 1,
        "deterministic": True,
        "example_count": len(examples),
        "dimensions": dimensions,
        "relationship_classes": classes,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------------------------

def save_training_examples(
    path: str = "data/alphabet_matrix_math.json",
    count: int = DEFAULT_EXAMPLE_COUNT,
) -> Dict[str, Any]:
  """
    Generate and save the deterministic matrix mathematics dataset.
    """

    examples = generate_training_examples(
        count=count
    )

    metadata = training_metadata(
        examples
    )

    payload = {
        "metadata": metadata,
        "examples": examples,
    }

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return metadata


# ---------------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------------

def load_training_examples(
    path: str = "data/alphabet_matrix_math.json",
) -> List[Dict[str, Any]]:
    """
    Load previously generated matrix mathematics examples.
    """

    if not os.path.exists(path):
        return []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(f)

    return payload.get(
        "examples",
        [],
    )


# ---------------------------------------------------------------------------
# BUILD PAIR WEIGHT MAP
# ---------------------------------------------------------------------------

def build_pair_weight_map(
    relationship_matrix: Dict[
        int,
        Sequence[str],
    ],
) -> Dict[str, float]:
    """
    Convert relationship classes into a direct pair -> weight mapping.

    Example:

        {
            "AB": 0.97,
            "CD": 0.94,
            ...
        }
    """

    result = {}

    for class_id, pairs in relationship_matrix.items():

        for position, pair in enumerate(pairs):

            pair = str(
                pair
            ).strip().upper()

            if len(pair) != 2:
                continue

            result[pair] = relationship_weight(
                class_id=class_id,
                pair_position=position,
            )

    return result


# ---------------------------------------------------------------------------
# WEIGHTED PAIR SCORE
# ---------------------------------------------------------------------------

def weighted_pair_score(
    pair: str,
    pair_weights: Dict[str, float],
) -> float:
    """
    Return the deterministic weight of an alphabet pair.
    """

    pair = str(
        pair
    ).strip().upper()

    if pair in pair_weights:
        return pair_weights[pair]

    reverse = pair[::-1]

    return pair_weights.get(
        reverse,
        0.0,
    )


# ---------------------------------------------------------------------------
# WORD MATRIX SCORE
# ---------------------------------------------------------------------------

def word_matrix_score(
    word: str,
    alphabet: str,
    matrix: torch.Tensor,
) -> float:
    """
    Produce a deterministic numerical relationship score for a word.

    Every adjacent alphabet pair contributes its corresponding matrix
    relationship value.

    The final result is normalized to 0.0 - 100.0.
    """

    word = str(
        word
    ).strip().lower()

    if len(word) < 2:
        return 0.0

    alphabet = str(
        alphabet
    ).lower()

    indices = {
        letter: index
        for index, letter
        in enumerate(alphabet)
    }

    scores = []

    for i in range(
        len(word) - 1
    ):

        first = word[i]
        second = word[i + 1]

        if (
            first not in indices
            or second not in indices
        ):
            continue

        score = matrix[
            indices[first],
            indices[second],
        ].item()

        scores.append(
            float(score)
        )

    if not scores:
        return 0.0

    return (
        sum(scores)
        / len(scores)
    ) * 100.0


# ---------------------------------------------------------------------------
# COMPOSITE MATRIX FEATURES
# ---------------------------------------------------------------------------

def matrix_features(
    word: str,
    alphabet: str,
    matrix: torch.Tensor,
) -> Dict[str, Any]:
    """
    Return numerical features that WordChain and Ranking can consume.
    """

    word = str(
        word
    ).strip().lower()

    score = word_matrix_score(
        word,
        alphabet,
        matrix,
    )

    pairs = []

    for i in range(
        len(word) - 1
    ):

        pair = word[
            i:i + 2
        ]

        pairs.append({
            "pair": pair,
            "weight": float(
                matrix[
                    alphabet.lower().find(pair[0]),
                    alphabet.lower().find(pair[1]),
                ].item()
            )
            if (
                pair[0] in alphabet.lower()
                and pair[1] in alphabet.lower()
            )
            else 0.0,
        })

    return {
        "word": word,
        "matrix_score": score,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

def matrix_math_registry(
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> Dict[str, Any]:
    """
    Return the complete deterministic mathematical registry.

    This registry is suitable for deployment metadata and later model
    synchronization.
    """

    examples = generate_training_examples(
        count=example_count
    )

    return {
        "module": "matrix_maths",
        "version": 1,
        "deterministic": True,
        "default_example_count": DEFAULT_EXAMPLE_COUNT,
        "operations": [
            "matrix_multiplication",
            "matrix_vector_multiplication",
            "relationship_propagation",
            "relationship_similarity",
            "weighted_relationship_score",
        ],
        "training": training_metadata(
            examples
        ),
    }


# ---------------------------------------------------------------------------
# DEVELOPMENT / VALIDATION
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "Generating deterministic matrix mathematics dataset..."
    )

    metadata = save_training_examples(
        path="data/alphabet_matrix_math.json",
        count=DEFAULT_EXAMPLE_COUNT,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    examples = load_training_examples(
        "data/alphabet_matrix_math.json"
    )

    print(
        f"Loaded {len(examples)} mathematical examples."
    )

    # Small demonstration matrix.
    demo = torch.tensor(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.6],
            [0.2, 0.6, 1.0],
        ],
        dtype=torch.float32,
    )

    vector = torch.tensor(
        [1.0, 0.5, 0.25],
        dtype=torch.float32,
    )

    result = matrix_vector_multiply(
        demo,
        vector,
    )

    print(
        "Matrix-vector result:",
        result.tolist(),
    )

    propagated = propagate_relationships(
        demo,
        vector,
        steps=2,
    )

    print(
        "Propagated relationship:",
        propagated.tolist(),
    )

    print(
        "Validation:",
        validate_training_examples(
            examples
        ),
    )
