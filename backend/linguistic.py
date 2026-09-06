"""
CoMpaNeoN Linguistic Layer
==========================

Multilingual linguistic analysis and enrichment layer.

ARCHITECTURE
------------

                    User / AI / External Text
                                │
                                ▼
                            langdetect
                                │
                                ▼
                           tokenizer.py
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
relation_and_alphabet_matrix.py              Token data
                │                                │
                ▼                                │
                         matrix_maths.py          │
                                │                │
                                └────────┬───────┘
                                         ▼
                                   linguistic.py
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
    Structural Analysis              Grammar                         Lexical
         │                               │                               │
         ├── vowels                     ├── parts of speech             ├── synonyms
         ├── consonants                 ├── conjunctions                ├── antonyms
         ├── syllables                  └── interjections               ├── semantic relationships
         ├── phonetic patterns                                              ├── hierarchical relationships
         ├── assonance                                                       └── close-proxy relationships
         └── resonance
                                         │
                                         ▼
                              Expression / Meaning
                                         │
                              ├── idioms
                              ├── figures of speech
                              ├── linguistic patterns
                              └── semantic analysis
                                         │
                                         ▼
                                  external.py
                                         │
                                         ▼
                               Dictionary enrichment
                                         │
                               definitions / meanings
                               synonyms / antonyms
                               examples / word forms
                               phonetics / etymology
                                         │
                                         ▼
                               Linguistic enrichment
                                         │
                                         ▼
                           Question Type Interpretation
                                         │
                                         ▼
                                WordUnderstanding
                                         │
                                         ▼
                                   WordChain


AUTHORITIES
-----------

tokenizer.py
    Owns canonical tokenization, language normalization,
    language alphabets and token coordinates.

relation_and_alphabet_matrix.py
    Owns alphabet/relationship substrate.

matrix_maths.py
    Owns mathematical signal construction.

external.py
    Owns communication with external dictionary providers.

question_type_detector.py
    Owns deterministic question-type interpretation.

word_understanding.py
    Owns structured understanding.

word_chain.py
    Owns word relationship, continuation and construction.

THIS MODULE
-----------

linguistic.py owns:

    - orchestration of linguistic members
    - structural linguistic analysis
    - internal lexical heuristics
    - grammar-oriented signal assembly
    - semantic relationship assembly
    - phonetic/orthographic signal analysis
    - external dictionary enrichment orchestration
    - unified linguistic output

IMPORTANT
---------

External dictionaries enrich CoMpaNeoN.

They do NOT replace CoMpaNeoN's internal linguistic intelligence.

This module does NOT:

    - place MemoryGrid documents
    - calculate GSP traversal
    - perform crawler traversal
    - rank candidates
    - generate prompts
    - generate AI responses
"""

from __future__ import annotations

import asyncio
import inspect
import re

from collections import Counter

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

try:

    from langdetect import (
        detect,
        LangDetectException,
    )

except ImportError:

    detect = None
    LangDetectException = Exception


# ============================================================================
# TOKENIZER AUTHORITY
# ============================================================================

from tokenizer import (
    tokenize,
    normalize_lang,
    alphabet_for,
)


# ============================================================================
# RELATION / ALPHABET MATRIX
# ============================================================================

try:

    from relation_and_alphabet_matrix import (
        analyze_relation_and_alphabet,
    )

except ImportError:

    analyze_relation_and_alphabet = None


# ============================================================================
# MATRIX MATHS
# ============================================================================

try:

    from matrix_maths import (
        intent_domain_signal,
        directive_signal,
        symbol_signal,
        code_language_signal,
        data_mixer_signal,
    )

except Exception:

    intent_domain_signal = None
    directive_signal = None
    symbol_signal = None
    code_language_signal = None
    data_mixer_signal = None


# ============================================================================
# QUESTION TYPE DETECTOR
# ============================================================================

try:

    from question_type_detector import (
        detect_question_type,
    )

except Exception:

    detect_question_type = None


# ============================================================================
# EXTERNAL DICTIONARY
# ============================================================================

try:

    from external import (
        fetch_dictionary,
    )

except Exception:

    fetch_dictionary = None


# ============================================================================
# BASIC LANGUAGE SETS
# ============================================================================

LATIN_VOWELS = set(
    "aeiou"
)

EXTENDED_LATIN_VOWELS = set(
    "aeiou"
    "àáâäãå"
    "æ"
    "èéêë"
    "ìíîï"
    "òóôöõ"
    "œ"
    "ùúûü"
    "ÿ"
    "ẹ"
    "ị"
    "ọ"
    "ụ"
    "ă"
    "ơ"
    "ư"
)

ARABIC_VOWEL_MARKS = set(
    "ًٌٍَُِّْ"
)

HEBREW_VOWEL_MARKS = set(
    "ְֱֲֳִֵֶַָֹֻּ"
)

DEVANAGARI_VOWELS = set(
    "अआइईउऊऋएऐओऔ"
)

BENGALI_VOWELS = set(
    "অআইঈউঊঋএঐওঔ"
)

JAPANESE_VOWELS = set(
    "あいうえお"
)

KOREAN_VOWEL_SIGNS = set(
    "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"
    "ㅐㅔㅚㅟㅢㅘㅙㅝㅞ"
)

THAI_VOWELS = set(
    "ะาิีึืุูเแโใไ"
)


# ============================================================================
# ENGLISH GRAMMAR HEURISTICS
# ============================================================================

ENGLISH_CONJUNCTIONS = {
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "although",
    "because",
    "since",
    "unless",
    "while",
    "whereas",
    "if",
    "though",
    "when",
    "before",
    "after",
}

ENGLISH_INTERJECTIONS = {
    "ah",
    "aha",
    "alas",
    "amen",
    "aw",
    "bravo",
    "eh",
    "hey",
    "hmm",
    "oh",
    "oops",
    "ouch",
    "ugh",
    "wow",
    "yay",
    "yes",
    "no",
}

ENGLISH_PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
}

ENGLISH_ARTICLES = {
    "a",
    "an",
    "the",
}

ENGLISH_PREPOSITIONS = {
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "to",
    "for",
    "of",
    "about",
    "under",
    "over",
    "between",
    "through",
    "into",
    "during",
    "before",
    "after",
}

ENGLISH_AUXILIARIES = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
}


# ============================================================================
# SEMANTIC RELATION LEXICON
# ============================================================================

RELATION_HINTS = {

    "cause": {
        "because",
        "therefore",
        "thus",
        "hence",
        "causes",
        "cause",
        "result",
        "results",
    },

    "comparison": {
        "than",
        "versus",
        "vs",
        "compare",
        "comparison",
        "difference",
        "similar",
    },

    "definition": {
        "means",
        "meaning",
        "defined",
        "definition",
        "called",
        "refers",
    },

    "hierarchy": {
        "type",
        "kind",
        "category",
        "class",
        "parent",
        "child",
        "subclass",
        "superclass",
    },

    "sequence": {
        "first",
        "then",
        "next",
        "after",
        "before",
        "finally",
        "last",
    },

    "condition": {
        "if",
        "unless",
        "when",
        "provided",
        "assuming",
    },

    "contrast": {
        "but",
        "however",
        "although",
        "whereas",
        "instead",
        "unlike",
    },
}


# ============================================================================
# IDIOM / FIGURATIVE MARKERS
# ============================================================================

COMMON_IDIOMS = {

    "break the ice",
    "piece of cake",
    "once in a blue moon",
    "under the weather",
    "spill the beans",
    "hit the nail on the head",
    "cost an arm and a leg",
    "beat around the bush",
    "burn the midnight oil",
    "the ball is in your court",
}

FIGURATIVE_MARKERS = {
    "like",
    "as",
    "as if",
    "as though",
    "literally",
    "metaphor",
    "symbolizes",
    "represents",
}


# ============================================================================
# TEXT HELPERS
# ============================================================================

WORD_RE = re.compile(
    r"[^\W\d_]+(?:[-'][^\W\d_]+)*",
    re.UNICODE,
)

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+"
)


def _clean_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def _words(
    text: str,
) -> List[str]:

    return [
        word
        for word in WORD_RE.findall(
            str(text or "").lower()
        )
        if word
    ]


def _safe_dict(
    value: Any,
) -> Dict[str, Any]:

    return (
        value
        if isinstance(value, dict)
        else {}
    )


# ============================================================================
# LANGUAGE RESOLUTION
# ============================================================================

def detect_language(
    text: str,
    fallback: str = "en",
) -> str:
    """
    Detect language, then pass the result through tokenizer.py.

    tokenizer.py remains the canonical language normalization authority.
    """

    clean = _clean_text(
        text
    )

    if not clean:
        return normalize_lang(
            fallback
        )

    if detect is None:
        return normalize_lang(
            fallback
        )

    try:

        detected = detect(
            clean
        )

        return normalize_lang(
            detected
        )

    except (
        LangDetectException,
        Exception,
    ):

        return normalize_lang(
            fallback
        )


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

def structural_analysis(
    text: str,
) -> Dict[str, Any]:

    clean = _clean_text(
        text
    )

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(
            clean
        )
        if sentence.strip()
    ]

    words = _words(
        clean
    )

    punctuation = [
        char
        for char in clean
        if not char.isalnum()
        and not char.isspace()
    ]

    return {

        "character_count":
            len(clean),

        "word_count":
            len(words),

        "sentence_count":
            len(sentences),

        "sentences":
            sentences,

        "words":
            words,

        "punctuation":
            punctuation,

        "punctuation_count":
            len(punctuation),

        "has_question":
            "?" in clean,

        "has_exclamation":
            "!" in clean,

        "has_statement":
            bool(clean)
            and not clean.endswith("?"),

        "token_density":
            (
                len(words)
                / max(
                    len(sentences),
                    1,
                )
            ),
    }


# ============================================================================
# VOWELS
# ============================================================================

def vowel_inventory(
    lang: str,
) -> Set[str]:

    language = normalize_lang(
        lang
    )

    if language in {
        "en",
        "fr",
        "de",
        "es",
        "pt",
        "tr",
        "it",
        "nl",
        "pl",
        "cs",
        "sv",
        "no",
        "da",
        "fi",
        "vi",
        "yo",
        "ha",
        "ig",
        "sw",
    }:

        return EXTENDED_LATIN_VOWELS

    if language == "ar":
        return ARABIC_VOWEL_MARKS

    if language == "he":
        return HEBREW_VOWEL_MARKS

    if language == "hi":
        return DEVANAGARI_VOWELS

    if language == "bn":
        return BENGALI_VOWELS

    if language == "ja":
        return JAPANESE_VOWELS

    if language == "ko":
        return KOREAN_VOWEL_SIGNS

    if language == "th":
        return THAI_VOWELS

    return EXTENDED_LATIN_VOWELS


def analyze_vowels(
    text: str,
    lang: str,
) -> Dict[str, Any]:

    clean = _clean_text(
        text
    )

    inventory = vowel_inventory(
        lang
    )

    found = [
        char
        for char in clean.lower()
        if char in inventory
    ]

    counts = Counter(
        found
    )

    return {

        "language":
            normalize_lang(lang),

        "inventory":
            sorted(inventory),

        "count":
            len(found),

        "unique":
            sorted(counts.keys()),

        "frequency":
            dict(counts),

        "positions":
            [
                index
                for index, char
                in enumerate(
                    clean.lower()
                )
                if char in inventory
            ],
    }


# ============================================================================
# CONSONANTS
# ============================================================================

def analyze_consonants(
    text: str,
    lang: str,
) -> Dict[str, Any]:

    clean = _clean_text(
        text
    )

    language = normalize_lang(
        lang
    )

    alphabet = set(
        alphabet_for(
            language
        )
    )

    vowels = vowel_inventory(
        language
    )

    found = [
        char
        for char in clean.lower()
        if char in alphabet
        and char not in vowels
    ]

    counts = Counter(
        found
    )

    return {

        "language":
            language,

        "count":
            len(found),

        "unique":
            sorted(counts.keys()),

        "frequency":
            dict(counts),

        "positions":
            [
                index
                for index, char
                in enumerate(
                    clean.lower()
                )
                if char in alphabet
                and char not in vowels
            ],
    }


# ============================================================================
# SYLLABIC STRUCTURE
# ============================================================================

def _estimate_latin_syllables(
    word: str,
) -> List[str]:

    clean = (
        word
        .lower()
        .strip()
    )

    if not clean:
        return []

    groups = re.findall(
        r"[aeiouy]+",
        clean,
    )

    if not groups:
        return [
            clean
        ]

    return groups


def analyze_syllabic_structure(
    text: str,
    lang: str,
) -> Dict[str, Any]:

    language = normalize_lang(
        lang
    )

    words = _words(
        text
    )

    per_word = []

    for word in words:

        if language in {
            "en",
            "fr",
            "de",
            "es",
            "pt",
            "tr",
            "it",
            "nl",
            "pl",
            "cs",
            "sv",
            "no",
            "da",
            "fi",
            "vi",
            "yo",
            "ha",
            "ig",
            "sw",
        }:

            units = _estimate_latin_syllables(
                word
            )

        else:

            units = [
                char
                for char in word
                if not char.isspace()
            ]

        per_word.append({

            "word":
                word,

            "units":
                units,

            "count":
                len(units),
        })

    return {

        "language":
            language,

        "words":
            per_word,

        "estimated_total":
            sum(
                item["count"]
                for item in per_word
            ),
    }


# ============================================================================
# PHONETIC STRUCTURE
# ============================================================================

def phonetic_structure(
    text: str,
    lang: str,
) -> Dict[str, Any]:

    language = normalize_lang(
        lang
    )

    vowels = vowel_inventory(
        language
    )

    words = _words(
        text
    )

    structures = []

    for word in words:

        pattern = []

        for char in word.lower():

            if char in vowels:
                pattern.append(
                    "V"
                )

            elif char.isalpha():
                pattern.append(
                    "C"
                )

            else:
                pattern.append(
                    "X"
                )

        structures.append({

            "word":
                word,

            "pattern":
                "".join(pattern),
        })

    return {

        "language":
            language,

        "structures":
            structures,
    }


# ============================================================================
# ASSONANCE
# ============================================================================

def analyze_assonance(
    text: str,
    lang: str,
) -> Dict[str, Any]:

 