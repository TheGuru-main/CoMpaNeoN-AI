"""
Relation and Alphabet Matrix

CoMpaNeoN relationship substrate.

The matrix contains 25 explicit relationship classes.
Each class contains alphabet-pair relationships.

This module does NOT:
    - replace tokenizer.py
    - replace word_understanding.py
    - replace ranking.py
    - replace memory_grid.py
    - generate prompts
    - generate AI responses

It provides deterministic alphabet/relationship information
to higher-level understanding and word-chain components.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# RELATIONSHIP MATRIX
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_letter(letter: str) -> str:
    return str(letter).strip().upper()


def normalize_pair(pair: str) -> str:
    return (
        str(pair)
        .strip()
        .upper()
    )


# ---------------------------------------------------------------------------
# CLASS LOOKUP
# ---------------------------------------------------------------------------

def get_relationship_class(
    pair: str,
) -> Optional[int]:

    pair = normalize_pair(pair)

    for class_id, pairs in RELATION_MATRIX.items():
        if pair in pairs:
            return class_id

    return None


# ---------------------------------------------------------------------------
# PAIRS FOR CLASS
# ---------------------------------------------------------------------------

def get_class_pairs(
    class_id: int,
) -> List[str]:

    return list(
        RELATION_MATRIX.get(
            int(class_id),
            [],
        )
    )


# ---------------------------------------------------------------------------
# LETTER RELATIONSHIP
# ---------------------------------------------------------------------------

def get_letter_relationships(
    letter: str,
) -> List[Dict[str, object]]:

    letter = normalize_letter(letter)

    relationships = []

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
                })

    return relationships


# ---------------------------------------------------------------------------
# WORD RELATIONSHIPS
# ---------------------------------------------------------------------------

def get_word_relationships(
    word: str,
) -> List[Dict[str, object]]:

    clean = (
        str(word)
        .strip()
        .upper()
    )

    relationships = []

    for letter in clean:

        relationships.extend(
            get_letter_relationships(
                letter
            )
        )

    return relationships


# ---------------------------------------------------------------------------
# PAIR EXTRACTION
# ---------------------------------------------------------------------------

def extract_alphabet_pairs(
    word: str,
) -> List[str]:

    clean = (
        str(word)
        .strip()
        .upper()
    )

    pairs = []

    for i in range(
        len(clean) - 1
    ):
        pairs.append(
            clean[i:i + 2]
        )

    return pairs


# ---------------------------------------------------------------------------
# WORD RELATIONSHIP CLASSES
# ---------------------------------------------------------------------------

def get_word_relationship_classes(
    word: str,
) -> List[int]:

    classes = []

    for pair in extract_alphabet_pairs(word):

        class_id = get_relationship_class(
            pair
        )

        if class_id is not None:
            classes.append(
                class_id
            )

    return classes


# ---------------------------------------------------------------------------
# RELATIONSHIP SIGNATURE
# ---------------------------------------------------------------------------

def relationship_signature(
    word: str,
) -> Tuple[int, ...]:

    return tuple(
        get_word_relationship_classes(
            word
        )
    )


# ---------------------------------------------------------------------------
# RELATED WORD SIGNAL
# ---------------------------------------------------------------------------

def relationship_score(
    word_a: str,
    word_b: str,
) -> float:

    sig_a = set(
        relationship_signature(
            word_a
        )
    )

    sig_b = set(
        relationship_signature(
            word_b
        )
    )

    if not sig_a or not sig_b:
        return 0.0

    intersection = sig_a & sig_b
    union = sig_a | sig_b

    return (
        len(intersection)
        / len(union)
    ) * 100.0