"""
Relation and Alphabet Matrix
===========================

CoMpaNeoN multilingual alphabet and relationship substrate.

Responsibilities
----------------
- Maintain the 25 deterministic relationship classes.
- Consume alphabets and language mapping from tokenizer.py.
- Provide language-aware letter indexing.
- Provide language-aware alphabet-pair relationships.
- Provide word relationship signatures.
- Provide alphabet matrix signatures.
- Provide relationship scores.
- Call matrix_maths.py for higher mathematical signal analysis.
- Call grid_cv.py through the established Matrix Maths signal bridge.
- Produce unified relation/alphabet signals for downstream
  linguistic, semantic and understanding layers.

Authority
---------
tokenizer.py
    Owns tokenization and language/alphabet mapping.

matrix_maths.py
    Owns mathematical signal construction.

grid_cv.py
    Owns grid comparison, validation and partition-compatible vectors.

This module
    Owns the relationship/alphabet substrate and connects those
    established signals without replacing their authorities.

This module does NOT:
    - tokenize text
    - perform GSP placement
    - traverse MemoryGrid
    - index MemoryGrid
    - replace GridCrawler
    - replace GridCV
    - replace ranking
    - generate prompts
    - generate AI responses
"""

from __future__ import annotations

import string

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langdetect import (
    detect,
    LangDetectException,
)


# ============================================================================
# OPTIONAL MATRIX MATHS CONNECTION
# ============================================================================

try:

    import matrix_maths

except Exception:

    matrix_maths = None


# ============================================================================
# LATIN RELATIONSHIP MATRIX
# ============================================================================

RELATION_MATRIX: Dict[int, List[str]] = {

    1: [
        "AB", "CD", "EF", "GH", "IJ", "KL", "MN",
        "OP", "QR", "ST", "UV", "WX", "YZ",
    ],

    2: [
        "BC", "DE", "FG", "HI", "JK", "LM", "NO",
        "PQ", "RS", "TU", "VZ", "WA", "XY",
    ],

    3: [
        "AZ", "BY", "CN", "DW", "EV", "FU", "GT",
        "HS", "IR", "JQ", "KP", "LO", "MX",
    ],

    4: [
        "ZB", "YC", "ND", "WE", "VF", "UG", "TH",
        "SI", "RJ", "QK", "PL", "OM", "XA",
    ],

    5: [
        "AC", "BD", "EH", "FI", "GJ", "KM", "LN",
        "OQ", "PR", "SU", "TV", "WY", "XZ",
    ],

    6: [
        "AN", "BO", "CP", "DQ", "ER", "FT", "GS",
        "HU", "IV", "JW", "KX", "LY", "MZ",
    ],

    7: [
        "PW", "JX", "HM", "DV", "BN", "QT", "FZ",
        "KS", "OY", "GL", "AU", "EI", "CR",
    ],

    8: [
        "VW", "OR", "BP", "MT", "HQ", "DG", "CK",
        "AY", "IX", "FS", "EN", "JZ", "LU",
    ],

    9: [
        "BT", "IN", "AS", "DZ", "JM", "EK", "CU",
        "HR", "VY", "OW", "QX", "FL", "GP",
    ],

    10: [
        "CS", "NW", "LQ", "FH", "OX", "PZ", "IY",
        "EU", "BJ", "MR", "GK", "AV", "DT",
    ],

    11: [
        "GN", "BW", "SZ", "AI", "FO", "LX", "DR",
        "MQ", "CE", "JU", "HY", "KV", "PT",
    ],

    12: [
        "JS", "NY", "FQ", "EX", "WZ", "KO", "BU",
        "IL", "HV", "CT", "GR", "AM", "DP",
    ],

    13: [
        "AH", "PU", "RW", "SX", "EM", "CL", "GI",
        "QV", "DY", "TZ", "JO", "BK", "FN",
    ],

    14: [
        "HP", "GQ", "BV", "CJ", "LS", "DM", "ET",
        "FY", "AK", "NX", "OU", "RZ", "IW",
    ],

    15: [
        "NR", "BE", "QW", "OS", "GZ", "LV", "FK",
        "HJ", "PY", "AT", "IM", "DU", "CX",
    ],

    16: [
        "HL", "KR", "GX", "SY", "IZ", "JN", "DF",
        "CV", "OT", "EQ", "UW", "AP", "BM",
    ],

    17: [
        "PV", "AO", "RY", "NU", "DL", "CW", "JT",
        "HX", "BS", "QZ", "FM", "EG", "IK",
    ],

    18: [
        "RX", "QY", "GV", "MU", "DS", "JP", "BF",
        "AL", "CI", "NZ", "KT", "EO", "HW",
    ],

    19: [
        "IP", "HK", "BQ", "CM", "AD", "GO", "JL",
        "RT", "VX", "UY", "NS", "EZ", "FW",
    ],

    20: [
        "FP", "JY", "BL", "DH", "IO", "NQ", "RV",
        "KU", "GW", "MS", "AE", "CZ", "TX",
    ],

    21: [
        "ES", "NP", "AJ", "FR", "UZ", "MV", "HO",
        "BI", "DX", "KW", "GY", "CQ", "LT",
    ],

    22: [
        "TY", "KZ", "AR", "HN", "QU", "EL", "MP",
        "BG", "FX", "JV", "DI", "SW", "CO",
    ],

    23: [
        "MW", "AG", "LZ", "DO", "RU", "KY", "NV",
        "FJ", "EP", "BX", "CH", "QS", "IT",
    ],

    24: [
        "IU", "BH", "OZ", "CF", "EY", "TW", "LR",
        "DJ", "PX", "AQ", "KN", "GM", "SV",
    ],

    25: [
        "EJ", "NT", "IQ", "AF", "OV", "BR", "HZ",
        "PS", "UX", "LW", "DK", "CG", "MY",
    ],
}


# ============================================================================
# TOKENIZER LANGUAGE BRIDGE
# ============================================================================

def _load_tokenizer_languages() -> Dict[str, Any]:
    """
    Discover language and alphabet definitions exposed by tokenizer.py.

    tokenizer.py remains the source of truth.
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

        value = getattr(
            tokenizer,
            name,
            None,
        )

        if isinstance(
            value,
            dict,
        ):
            return value

    return {}


TOKENIZER_LANGUAGES = (
    _load_tokenizer_languages()
)


# ============================================================================
# LANGUAGE NORMALIZATION
# ============================================================================

def normalize_language(
    lang: Optional[str],
    fallback: str = "en",
) -> str:

    value = (
        str(
            lang
            or fallback
        )
        .strip()
        .lower()
    )

    try:

        import tokenizer

        normalizer = getattr(
            tokenizer,
            "normalize_lang",
            None,
        )

        if callable(
            normalizer
        ):

            return str(
                normalizer(
                    value
                )
            ).strip().lower()

    except Exception:

        pass

    return value or fallback


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(
    text: str,
    fallback: str = "en",
) -> str:
    """
    Detect language.

    tokenizer.py normalization is used after detection where available.
    """

    clean = str(
        text
        or ""
    ).strip()

    if not clean:

        return normalize_language(
            fallback
        )

    try:

        detected = detect(
            clean
        )

        return normalize_language(
            detected,
            fallback=fallback,
        )

    except LangDetectException:

        return normalize_language(
            fallback
        )

    except Exception:

        return normalize_language(
            fallback
        )


# ============================================================================
# ALPHABET EXTRACTION
# ============================================================================

def _extract_alphabet(
    language_config: Any,
) -> Optional[str]:

    if isinstance(
        language_config,
        str,
    ):

        return language_config

    if isinstance(
        language_config,
        dict,
    ):

        for key in (
            "alphabet",
            "letters",
            "chars",
            "characters",
        ):

            value = language_config.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                return value

            if isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            ):

                return "".join(
                    str(item)
                    for item in value
                )

    return None


# ============================================================================
# UNIQUE ALPHABET CHARACTERS
# ============================================================================

def _unique_characters(
    value: str,
) -> str:

    seen = set()

    result = []

    for char in str(
        value
    ):

        if char in seen:
            continue

        seen.add(
            char
        )

        result.append(
            char
        )

    return "".join(
        result
    )


# ============================================================================
# LANGUAGE ALPHABET
# ============================================================================

def get_language_alphabet(
    lang: str,
) -> str:
    """
    Return the alphabet defined by tokenizer.py.

    English retains the standard Latin fallback.
    """

    normalized = normalize_language(
        lang
    )

    config = (
        TOKENIZER_LANGUAGES.get(
            normalized
        )
    )

    alphabet = _extract_alphabet(
        config
    )

    if alphabet:

        return _unique_characters(
            alphabet
        )

    if normalized in {
        "en",
        "eng",
        "english",
    }:

        return string.ascii_lowercase

    return string.ascii_lowercase


# ============================================================================
# SUPPORTED LANGUAGES
# ============================================================================

def supported_languages() -> List[str]:

    return sorted(
        str(
            lang
        )
        for lang in TOKENIZER_LANGUAGES.keys()
    )


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_letter(
    letter: str,
) -> str:

    return str(
        letter
    ).strip().upper()


def normalize_pair(
    pair: str,
) -> str:

    return str(
        pair
    ).strip().upper()


# ============================================================================
# LETTER INDEX
# ============================================================================

def letter_index(
    letter: str,
    lang: str = "en",
) -> Optional[int]:

    clean = normalize_letter(
        letter
    )

    if not clean:
        return None

    alphabet = (
        get_language_alphabet(
            lang
        ).upper()
    )

    try:

        return alphabet.index(
            clean
        )

    except ValueError:

        return None


# ============================================================================
# RELATIONSHIP CLASS LOOKUP
# ============================================================================

def get_relationship_class(
    pair: str,
) -> Optional[int]:

    normalized = normalize_pair(
        pair
    )

    for class_id, pairs in (
        RELATION_MATRIX.items()
    ):

        if normalized in pairs:

            return class_id

    return None


# ============================================================================
# LANGUAGE-AWARE RELATIONSHIP LOOKUP
# ============================================================================

def get_relationship_class_for_letters(
    letter_a: str,
    letter_b: str,
    lang: str = "en",
) -> Optional[int]:

    a = normalize_letter(
        letter_a
    )

    b = normalize_letter(
        letter_b
    )

    if not a or not b:

        return None

    direct = (
        get_relationship_class(
            f"{a}{b}"
        )
    )

    if direct is not None:

        return direct

    return get_relationship_class(
        f"{b}{a}"
    )


# ============================================================================
# CLASS PAIRS
# ============================================================================

def get_class_pairs(
    class_id: int,
) -> List[str]:

    return list(
        RELATION_MATRIX.get(
            int(
                class_id
            ),
            [],
        )
    )


# ============================================================================
# ALPHABET MATRIX
# ============================================================================

def build_alphabet_matrix(
    lang: str,
) -> List[List[float]]:
    """
    Build the local deterministic alphabet relationship matrix.

    Matrix Maths can provide weighted mathematical versions of this
    substrate while this module retains the canonical relationship view.
    """

    alphabet = (
        get_language_alphabet(
            lang
        )
    )

    size = len(
        alphabet
    )

    matrix = [
        [
            0.0
            for _ in range(
                size
            )
        ]
        for _ in range(
            size
        )
    ]

    position = {
        char: index
        for index, char in enumerate(
            alphabet.upper()
        )
    }

    latin = (
        string.ascii_uppercase
    )

    if set(
        latin
    ).issubset(
        position
    ):

        for pairs in (
            RELATION_MATRIX.values()
        ):

            for pair in pairs:

                a = pair[0]
                b = pair[1]

                if (
                    a not in position
                    or b not in position
                ):
                    continue

                i = position[a]
                j = position[b]

                if i == j:
                    continue

                matrix[i][j] = 1.0
                matrix[j][i] = 1.0

    return matrix


# ============================================================================
# ALL LANGUAGE MATRICES
# ============================================================================

def build_all_alphabet_matrices() -> Dict[
    str,
    List[List[float]],
]:

    return {
        lang: build_alphabet_matrix(
            lang
        )
        for lang in supported_languages()
    }


# ============================================================================
# LETTER RELATIONSHIPS
# ============================================================================

def get_letter_relationships(
    letter: str,
    lang: str = "en",
) -> List[Dict[str, Any]]:

    language = normalize_language(
        lang
    )

    clean = normalize_letter(
        letter
    )

    relationships: List[
        Dict[str, Any]
    ] = []

    for class_id, pairs in (
        RELATION_MATRIX.items()
    ):

        for pair in pairs:

            if clean not in pair:
                continue

            other = (
                pair[1]
                if pair[0] == clean
                else pair[0]
            )

            relationships.append(
                {
                    "class": class_id,
                    "pair": pair,
                    "letter": clean,
                    "related_letter": other,
                    "language": language,
                    "source": (
                        "relationship_matrix"
                    ),
                }
            )

    index = letter_index(
        clean,
        language,
    )

    if index is None:

        return relationships

    alphabet = (
        get_language_alphabet(
            language
        ).upper()
    )

    matrix = (
        build_alphabet_matrix(
            language
        )
    )

    if index >= len(
        matrix
    ):

        return relationships

    for other_index, connected in enumerate(
        matrix[index]
    ):

        if not connected:
            continue

        if (
            other_index
            >= len(
                alphabet
            )
        ):
            continue

        other = (
            alphabet[
                other_index
            ]
        )

        relationships.append(
            {
                "class": (
                    get_relationship_class_for_letters(
                        clean,
                        other,
                        language,
                    )
                ),
                "pair": (
                    f"{clean}{other}"
                ),
                "letter": clean,
                "related_letter": other,
                "language": language,
                "matrix": True,
                "source": (
                    "alphabet_matrix"
                ),
            }
        )

    return relationships


# ============================================================================
# WORD RELATIONSHIPS
# ============================================================================

def get_word_relationships(
    word: str,
    lang: str = "en",
) -> List[Dict[str, Any]]:

    language = normalize_language(
        lang
    )

    clean = str(
        word
        or ""
    ).strip()

    relationships: List[
        Dict[str, Any]
    ] = []

    for letter in clean:

        relationships.extend(
            get_letter_relationships(
                letter,
                lang=language,
            )
        )

    return relationships


# ============================================================================
# PAIR EXTRACTION
# ============================================================================

def extract_alphabet_pairs(
    word: str,
) -> List[str]:

    clean = str(
        word
        or ""
    ).strip().upper()

    return [
        clean[
            index:index + 2
        ]
        for index in range(
            max(
                0,
                len(clean) - 1,
            )
        )
    ]


# ============================================================================
# WORD RELATIONSHIP CLASSES
# ============================================================================

def get_word_relationship_classes(
    word: str,
    lang: str = "en",
) -> List[int]:

    classes: List[int] = []

    for pair in (
        extract_alphabet_pairs(
            word
        )
    ):

        if len(pair) != 2:
            continue

        class_id = (
            get_relationship_class_for_letters(
                pair[0],
                pair[1],
                lang=lang,
            )
        )

        if class_id is not None:

            classes.append(
                class_id
            )

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
# ALPHABET MATRIX SIGNATURE
# ============================================================================

def alphabet_matrix_signature(
    word: str,
    lang: str = "en",
) -> Tuple[float, ...]:

    clean = str(
        word
        or ""
    ).strip()

    if len(clean) < 2:

        return ()

    language = normalize_language(
        lang
    )

    matrix = (
        build_alphabet_matrix(
            language
        )
    )

    result: List[
        float
    ] = []

    for index in range(
        len(clean) - 1
    ):

        a = letter_index(
            clean[index],
            language,
        )

        b = letter_index(
            clean[index + 1],
            language,
        )

        if (
            a is None
            or b is None
        ):

            result.append(
                0.0
  