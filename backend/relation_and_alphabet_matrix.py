"""
Alphabet Matrix
================

CoMpaNeoN multilingual alphabet/relationship substrate.

Responsibilities:
- Maintain the 25 deterministic relationship classes.
- Map every alphabet exposed by tokenizer.py.
- Provide language-aware letter indexing.
- Provide language-aware alphabet-pair relationships.
- Provide word relationship signatures.
- Provide relationship scores for WordChain and WordUnderstanding.
- Provide deterministic matrix representations suitable for
  downstream matrix operations / deep-learning infrastructure.

Important:
- tokenizer.py remains the authority for language/token definitions.
- This module does NOT tokenize text.
- This module does NOT replace ranking.py.
- This module does NOT replace memory_grid.py.
- This module does NOT generate responses.

Architecture:

    tokenizer.py
          ↓
    alphabet_matrix.py
          ↓
    ranking.py
          ↓
    word_chain.py
          ↓
    word_understanding.py
"""

from __future__ import annotations
from langdetect import detect, LangDetectException
from typing import Any, Dict, Iterable, List, Optional, Tuple

import string


# ============================================================================
# LATIN RELATIONSHIP MATRIX
# ============================================================================

RELATION_MATRIX: Dict[int, List[str]] = {

    1: [
        "AB", "CD", "EF", "GH", "IJ", "KL", "MN",
        "OP", "QR", "ST", "UV", "WX", "YZ"
    ],

    2: [
        "BC", "DE", "FG", "HI", "JK", "LM", "NO",
        "PQ", "RS", "TU", "VZ", "WA", "XY"
    ],

    3: [
        "AZ", "BY", "CN", "DW", "EV", "FU", "GT",
        "HS", "IR", "JQ", "KP", "LO", "MX"
    ],

    4: [
        "ZB", "YC", "ND", "WE", "VF", "UG", "TH",
        "SI", "RJ", "QK", "PL", "OM", "XA"
    ],

    5: [
        "AC", "BD", "EH", "FI", "GJ", "KM", "LN",
        "OQ", "PR", "SU", "TV", "WY", "XZ"
    ],

    6: [
        "AN", "BO", "CP", "DQ", "ER", "FT", "GS",
        "HU", "IV", "JW", "KX", "LY", "MZ"
    ],

    7: [
        "PW", "JX", "HM", "DV", "BN", "QT", "FZ",
        "KS", "OY", "GL", "AU", "EI", "CR"
    ],

    8: [
        "VW", "OR", "BP", "MT", "HQ", "DG", "CK",
        "AY", "IX", "FS", "EN", "JZ", "LU"
    ],

    9: [
        "BT", "IN", "AS", "DZ", "JM", "EK", "CU",
        "HR", "VY", "OW", "QX", "FL", "GP"
    ],

    10: [
        "CS", "NW", "LQ", "FH", "OX", "PZ", "IY",
        "EU", "BJ", "MR", "GK", "AV", "DT"
    ],

    11: [
        "GN", "BW", "SZ", "AI", "FO", "LX", "DR",
        "MQ", "CE", "JU", "HY", "KV", "PT"
    ],

    12: [
        "JS", "NY", "FQ", "EX", "WZ", "KO", "BU",
        "IL", "HV", "CT", "GR", "AM", "DP"
    ],

    13: [
        "AH", "PU", "RW", "SX", "EM", "CL", "GI",
        "QV", "DY", "TZ", "JO", "BK", "FN"
    ],

    14: [
        "HP", "GQ", "BV", "CJ", "LS", "DM", "ET",
        "FY", "AK", "NX", "OU", "RZ", "IW"
    ],

    15: [
        "NR", "BE", "QW", "OS", "GZ", "LV", "FK",
        "HJ", "PY", "AT", "IM", "DU", "CX"
    ],

    16: [
        "HL", "KR", "GX", "SY", "IZ", "JN", "DF",
        "CV", "OT", "EQ", "UW", "AP", "BM"
    ],

    17: [
        "PV", "AO", "RY", "NU", "DL", "CW", "JT",
        "HX", "BS", "QZ", "FM", "EG", "IK"
    ],

    18: [
        "RX", "QY", "GV", "MU", "DS", "JP", "BF",
        "AL", "CI", "NZ", "KT", "EO", "HW"
    ],

    19: [
        "IP", "HK", "BQ", "CM", "AD", "GO", "JL",
        "RT", "VX", "UY", "NS", "EZ", "FW"
    ],

    20: [
        "FP", "JY", "BL", "DH", "IO", "NQ", "RV",
        "KU", "GW", "MS", "AE", "CZ", "TX"
    ],

    21: [
        "ES", "NP", "AJ", "FR", "UZ", "MV", "HO",
        "BI", "DX", "KW", "GY", "CQ", "LT"
    ],

    22: [
        "TY", "KZ", "AR", "HN", "QU", "EL", "MP",
        "BG", "FX", "JV", "DI", "SW", "CO"
    ],

    23: [
        "MW", "AG", "LZ", "DO", "RU", "KY", "NV",
        "FJ", "EP", "BX", "CH", "QS", "IT"
    ],

    24: [
        "IU", "BH", "OZ", "CF", "EY", "TW", "LR",
        "DJ", "PX", "AQ", "KN", "GM", "SV"
    ],

    25: [
        "EJ", "NT", "IQ", "AF", "OV", "BR", "HZ",
        "PS", "UX", "LW", "DK", "CG", "MY"
    ],
}


# ============================================================================
# TOKENIZER LANGUAGE BRIDGE
# ============================================================================

def _load_tokenizer_languages() -> Dict[str, Any]:
    """
    Discover language/alphabet definitions exposed by tokenizer.py.

    tokenizer.py remains the source of truth.

    The bridge intentionally supports several possible exported names so
    alphabet_matrix.py does not duplicate tokenizer configuration.
    """

    try:
        import tokenizer
    except ImportError:
        return {}

    candidates = (
        "LANGUAGE_ALPHABETS",
        "LANG_ALPHABETS",
        "ALPHABETS",
        "LANGUAGE_CONFIG",
        "LANGUAGES",
    )

    for name in candidates:
        value = getattr(tokenizer, name, None)

        if isinstance(value, dict):
            return value

    return {}


TOKENIZER_LANGUAGES = _load_tokenizer_languages()


# ============================================================================
# ALPHABET REGISTRY
# ============================================================================

def _extract_alphabet(
    language_config: Any,
) -> Optional[str]:
    """
    Extract an alphabet from a tokenizer language definition.

    Supported forms:

        "abcdefghijklmnopqrstuvwxyz"

    or:

        {
            "alphabet": "...",
        }

    or:

        {
            "letters": "...",
        }
    """

    if isinstance(language_config, str):
        return language_config

    if isinstance(language_config, dict):

        for key in (
            "alphabet",
            "letters",
            "chars",
            "characters",
        ):

            value = language_config.get(key)

            if isinstance(value, str):
                return value

            if isinstance(value, (list, tuple)):
                return "".join(
                    str(x)
                    for x in value
                )

    return None


def get_language_alphabet(
    lang: str,
) -> str:
    """
    Return the tokenizer-defined alphabet for a language.

    English falls back to the standard Latin alphabet when tokenizer.py
    does not expose a separate alphabet definition.
    """

    normalized = str(lang).strip().lower()

    config = TOKENIZER_LANGUAGES.get(normalized)

    alphabet = _extract_alphabet(config)

    if alphabet:
        return _unique_characters(alphabet)

    if normalized in {
        "en",
        "eng",
        "english",
    }:
        return string.ascii_lowercase

    return string.ascii_lowercase


def _unique_characters(
    value: str,
) -> str:
    """
    Preserve alphabet order while removing duplicate characters.
    """

    seen = set()
    result = []

    for char in str(value):

        if char in seen:
            continue

        seen.add(char)
        result.append(char)

    return "".join(result)


def supported_languages() -> List[str]:
    """
    Return every language currently exposed by tokenizer.py.
    """

    return sorted(
        str(lang)
        for lang in TOKENIZER_LANGUAGES.keys()
    )


# ============================================================================
# LANGUAGE ALPHABET MATRIX
# ============================================================================

def build_alphabet_matrix(
    lang: str,
) -> List[List[int]]:
    """
    Build a deterministic alphabet adjacency matrix.

    Matrix[i][j] = 1 when the two alphabet positions participate in
    any of the 25 relationship classes.

    The matrix is symmetric because alphabet relationships are undirected.
    """

    alphabet = get_language_alphabet(lang)

    size = len(alphabet)

    matrix = [
        [0 for _ in range(size)]
        for _ in range(size)
    ]

    # ------------------------------------------------------------------
    # Latin relationship classes can be projected directly.
    # ------------------------------------------------------------------

    latin = string.ascii_uppercase

    position = {
        char: index
        for index, char in enumerate(alphabet.upper())
    }

    if set(latin).issubset(position):

        for pairs in RELATION_MATRIX.values():

            for pair in pairs:

                a = pair[0]
                b = pair[1]

                if a not in position:
                    continue

                if b not in position:
                    continue

                i = position[a]
                j = position[b]

                if i == j:
                    continue

                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


# ============================================================================
# LANGUAGE RELATIONSHIP MATRICES
# ============================================================================

def build_all_alphabet_matrices() -> Dict[str, List[List[int]]]:
    """
    Build the alphabet matrix for every tokenizer-supported language.
    """

    return {
        lang: build_alphabet_matrix(lang)
        for lang in supported_languages()
    }


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_letter(
    letter: str,
) -> str:
    return str(letter).strip().upper()


def normalize_pair(
    pair: str,
) -> str:
    return str(pair).strip().upper()


# ============================================================================
# LETTER INDEX
# ============================================================================

def letter_index(
    letter: str,
    lang: str = "en",
) -> Optional[int]:
    """
    Return deterministic alphabet position for a letter.

    Index is zero-based internally.
    """

    clean = normalize_letter(letter)

    if not clean:
        return None

    alphabet = get_language_alphabet(lang).upper()

    try:
        return alphabet.index(clean)
    except ValueError:
        return None


# ============================================================================
# RELATIONSHIP CLASS LOOKUP
# ============================================================================

def get_relationship_class(
    pair: str,
) -> Optional[int]:

    pair = normalize_pair(pair)

    for class_id, pairs in RELATION_MATRIX.items():

        if pair in pairs:
            return class_id

    return None


# ============================================================================
# RELATIONSHIP CLASS LOOKUP
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect the language of a text using langdetect.

    Falls back to English for empty, very short, ambiguous,
    or otherwise undetectable text.
    """

    if not text or not str(text).strip():
        return "en"

    try:
        return detect(str(text))
    except LangDetectException:
        return "en"
    except Exception:
        return "en"

# ============================================================================
# LANGUAGE-AWARE RELATIONSHIP LOOKUP
# ============================================================================

def get_relationship_class_for_letters(
    letter_a: str,
    letter_b: str,
    lang: str = "en",
) -> Optional[int]:
    """
    Resolve the relationship class between two letters.

    For the Latin alphabet this uses the existing 25 classes.

    For non-Latin alphabets, the alphabet matrix remains available even
    when no Latin class mapping exists.
    """

    a = normalize_letter(letter_a)
    b = normalize_letter(letter_b)

    if not a or not b:
        return None

    pair = f"{a}{b}"

    direct = get_relationship_class(pair)

    if direct is not None:
        return direct

    reverse = get_relationship_class(
        f"{b}{a}"
    )

    if reverse is not None:
        return reverse

    return None


# ============================================================================
# CLASS PAIRS
# ============================================================================

def get_class_pairs(
    class_id: int,
) -> List[str]:

    return list(
        RELATION_MATRIX.get(
            int(class_id),
            [],
        )
    )


# ============================================================================
# LETTER RELATIONSHIPS
# ============================================================================

def get_letter_relationships(
    letter: str,
    lang: str = "en",
) -> List[Dict[str, object]]:

    letter = normalize_letter(letter)

    relationships = []

    alphabet = get_language_alphabet(lang).upper()

    # ---------------------------------------------------------------
    # Existing 25-class relationships.
    # ---------------------------------------------------------------

    for class_id, pairs in RELATION_MATRIX.items():

        for pair in pairs:

            if letter in pair:

                other = (
                    pair[1]
                    if pair[0] == letter
                    else pair[0]
                )

                relationships.append({
                    "class": class_id,
                    "pair": pair,
                    "letter": letter,
                    "related_letter": other,
                    "language": lang,
                })

    # ---------------------------------------------------------------
    # Matrix relationships.
    #
    # This is important for languages whose alphabet is not A-Z.
    # ---------------------------------------------------------------

    index = letter_index(
        letter,
        lang,
    )

    if index is not None:

        matrix = build_alphabet_matrix(
            lang
        )

        for other_index, connected in enumerate(
            matrix[index]
        ):

            if not connected:
                continue

            if other_index >= len(alphabet):
                continue

            other = alphabet[other_index]

            relationships.append({
                "class": None,
                "pair": f"{letter}{other}",
                "letter": letter,
                "related_letter": other,
                "language": lang,
                "matrix": True,
            })

    return relationships


# ============================================================================
# WORD RELATIONSHIPS
# ============================================================================

def get_word_relationships(
    word: str,
    lang: str = "en",
) -> List[Dict[str, object]]:

    clean = str(word).strip()

    relationships = []

    for letter in clean:

        relationships.extend(
            get_letter_relationships(
                letter,
                lang=lang,
            )
        )

    return relationships


# ============================================================================
# PAIR EXTRACTION
# ============================================================================

def extract_alphabet_pairs(
    word: str,
) -> List[str]:

    clean = str(word).strip().upper()

    return [
        clean[i:i + 2]
        for i in range(len(clean) - 1)
    ]


# ============================================================================
# WORD RELATIONSHIP CLASSES
# ============================================================================

def get_word_relationship_classes(
    word: str,
    lang: str = "en",
) -> List[int]:

    classes = []

    for pair in extract_alphabet_pairs(word):

        class_id = get_relationship_class_for_letters(
            pair[0],
            pair[1],
            lang=lang,
        )

        if class_id is not None:
            classes.append(class_id)

    return classes


# ============================================================================
# RELATIONSHIP SIGNATURE
# ============================================================================

def relationship_signature(
    word: str,
    lang: str = "en",
) -> Tuple[int, ...]:

    return tuple(
        get_word_relationship_classes(
            word,
            lang=lang,
        )
    )


# ============================================================================
# MATRIX SIGNATURE
# ============================================================================

def alphabet_matrix_signature(
    word: str,
    lang: str = "en",
) -> Tuple[int, ...]:
    """
    Encode a word as a sequence of alphabet-matrix relationship signals.

    1 = related letters
    0 = unrelated letters
    """

    clean = str(word).strip()

    if len(clean) < 2:
        return ()

    matrix = build_alphabet_matrix(lang)

    result = []

    for i in range(len(clean) - 1):

        a = letter_index(
            clean[i],
            lang,
        )

        b = letter_index(
            clean[i + 1],
            lang,
        )

        if a is None or b is None:
            result.append(0)
            continue

        if (
            a >= len(matrix)
            or b >= len(matrix)
        ):
            result.append(0)
            continue

        result.append(
            matrix[a][b]
        )

    return tuple(result)


# ============================================================================
# RELATED WORD SIGNAL
# ============================================================================

def relationship_score(
    word_a: str,
    word_b: str,
    lang: str = "en",
) -> float:

    sig_a = set(
        relationship_signature(
            word_a,
            lang=lang,
        )
    )

    sig_b = set(
        relationship_signature(
            word_b,
            lang=lang,
        )
    )

    if not sig_a or not sig_b:
        return 0.0

    intersection = sig_a & sig_b
    union = sig_a | sig_b

    if not union:
        return 0.0

    return (
        len(intersection)
        / len(union)
    ) * 100.0


# ============================================================================
# MATRIX RELATIONSHIP SCORE
# ============================================================================

def alphabet_matrix_score(
    word_a: str,
    word_b: str,
    lang: str = "en",
) -> float:
    """
    Compare two words through their alphabet-matrix signatures.
    """

    sig_a = alphabet_matrix_signature(
        word_a,
        lang,
    )

    sig_b = alphabet_matrix_signature(
        word_b,
        lang,
    )

    if not sig_a or not sig_b:
        return 0.0

    size = min(
        len(sig_a),
        len(sig_b),
    )

    if size == 0:
        return 0.0

    matches = sum(
        1
        for a, b in zip(
            sig_a[:size],
            sig_b[:size],
        )
        if a == b
    )

    return (
        matches / size
    ) * 100.0


# ============================================================================
# RELATIONSHIP FEATURE VECTOR
# ============================================================================

def relationship_feature_vector(
    word: str,
    lang: str = "en",
) -> List[float]:
    """
    Produce a deterministic feature vector suitable for WordChain and
    future matrix/deep-learning operations.

    Layout:

        [25 relationship-class features]
        +
        [alphabet transition features]
    """

    classes = relationship_signature(
        word,
        lang=lang,
    )

    class_features = [
        0.0
        for _ in range(25)
    ]

    for class_id in classes:

        if 1 <= class_id <= 25:
            class_features[class_id - 1] = 1.0

    matrix_features = [
        float(x)
        for x in alphabet_matrix_signature(
            word,
            lang=lang,
        )
    ]

    return (
        class_features
        + matrix_features
    )


# ============================================================================
# LANGUAGE INFORMATION
# ============================================================================

def language_info(
    lang: str,
) -> Dict[str, Any]:

    alphabet = get_language_alphabet(lang)

    return {
        "language": str(lang).lower(),
        "alphabet": alphabet,
        "alphabet_size": len(alphabet),
        "matrix_size": [
            len(alphabet),
            len(alphabet),
        ],
    }


# ============================================================================
# EXPORT
# ============================================================================

def export_matrix_registry() -> Dict[str, Any]:
    """
    Return the complete deterministic alphabet-matrix registry.

    This is useful when the matrix later becomes part of the model
    training/deployment metadata.
    """

    return {
        "relationship_classes": {
            str(class_id): list(pairs)
            for class_id, pairs
            in RELATION_MATRIX.items()
        },
        "languages": {
            lang: language_info(lang)
            for lang in supported_languages()
        },
    }