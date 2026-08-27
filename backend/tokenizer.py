"""
CoMpaNeoN Tokenizer
===================

Multilingual tokenizer and alphabet-grid foundation.

Responsibilities:
- Language normalization.
- Multilingual alphabet definitions.
- Multilingual keyboard/input key lines.
- Letter-grid indexing.
- Word-grid indexing.
- Stem prefixes/affixes.
- Accent/code-mix bridging.
- Global symbols board.
- Deterministic token construction.
- Lexical letter scoring.
- Lexical word scoring.

IMPORTANT
---------

The tokenizer does NOT own:

    GSP storage placement
    GSP XOR sharding
    MemoryGrid storage
    Full-text K replication
    Ranking
    WordChain
    Follow-up generation
    Prompt management
    AI response generation

GSP keyboard calculations are delegated to keyboard.py.

GRID RULES
----------

Letter Grid:

    Alphabet index is reduced with R = 1.

    letter_index % 1

Word Grid:

    Word index is reduced with R = 26.

    word_index % 26

FIRST LETTER c
--------------

    c = first_letter_index

c is CONSTANT for the token.

It is NOT:

    c % A

K
-

K is NOT applied here.

K belongs to full-text placement and MemoryGrid.

GSP START ROW
-------------

The canonical GSP start-row formula remains in keyboard.py:

    start_row = ((Lsum + Ssum - 1) % R) + 1

PERTURBATION
------------

Full-text perturbation belongs to the GSP/MemoryGrid layer.

Configured values:

    forward_d  = 5
    backward_d = 1

The tokenizer does not apply these perturbations.
"""

from __future__ import annotations

import re
from typing import Any

import keyboard


# =====================================================================
# MULTILINGUAL KEY MAPPINGS
# =====================================================================

EN_KEY_LINE = "QWERTYUIOPASDFGHJKLZXCVBNM"

FR_KEY_LINE = "AZERTYUIOPQSDFGHJKLMWXCVBN"

DE_KEY_LINE = "QWERTZUIOPASDFGHJKLYXCVBNM"

AR_KEY_LINE = "ضصثقفغعهخحجدشسيبلاتنمكطئءؤرلاىةوزظ"

HE_KEY_LINE = "קראטוןםפשדגכעיחלךףזסבהנמצתץ"

EL_KEY_LINE = ";ςερτυθιοπασδφγηξκλζχψωβνμ"

RU_KEY_LINE = "йцукенгшщзхъфывапролджэячсмитьбю"

UK_KEY_LINE = "йцукенгшщзхїфівапролдэжячсмитьбю"

HI_KEY_LINE = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"

BN_KEY_LINE = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"

JA_HIRAGANA_KEY_LINE = (
    "あいうえおかきくけこさしすせそ"
    "たちつてとなにぬねのはひふへほ"
    "まみむめもやゆよらりるれろわをん"
)

KO_HANGUL_KEY_LINE = (
    "ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ"
    "ㅁㄴㅇㄹㅎㅗㅓㅏㅣ"
    "ㅋㅌㅊㅍㅠㅜㅡ"
)

ZH_PINYIN_KEY_LINE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

TR_KEY_LINE = "QWERTYUIOPĞÜASDFGHJKLŞİZXCVBNMÖÇ"

ES_KEY_LINE = "QWERTYUIOPASDFGHJKLÑZXCVBNM"

IT_KEY_LINE = "QWERTYUIOPÈASDFGHJKLÒÀZXCVBNM"

PT_KEY_LINE = "QWERTYUIOP´ASDFGHJKLÇ~ZXCVBNM"

NL_KEY_LINE = "QWERTYUIOPASDFGHJKLZXCVBNM"

PL_KEY_LINE = "QWERTYUIOPĄASDFGHJKLŚZXCVBNMĘ"

CS_KEY_LINE = "QWERTZUIOPÚASDFGHJKLÝZXCVBNM"

SV_KEY_LINE = "QWERTYUIOPÅASDFGHJKLÖZXCVBNM"

NO_KEY_LINE = "QWERTYUIOPÅASDFGHJKLØZXCVBNM"

DA_KEY_LINE = "QWERTYUIOPÅASDFGHJKLÆZXCVBNM"

FI_KEY_LINE = "QWERTYUIOPÅASDFGHJKLÖZXCVBNM"

VI_KEY_LINE = "QWERTYUIOPASDFGHJKLZXCVBNMĐ"

TH_KEY_LINE = (
    "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธน"
    "บปผฝพฟภมยรฤลฦวศษสหฬอฮ"
)


# =====================================================================
# KEY-LINE REGISTRY
# =====================================================================

KEY_LINES: dict[str, str] = {
    "en": EN_KEY_LINE,
    "fr": FR_KEY_LINE,
    "de": DE_KEY_LINE,
    "ar": AR_KEY_LINE,
    "he": HE_KEY_LINE,
    "el": EL_KEY_LINE,
    "ru": RU_KEY_LINE,
    "uk": UK_KEY_LINE,
    "hi": HI_KEY_LINE,
    "bn": BN_KEY_LINE,
    "ja": JA_HIRAGANA_KEY_LINE,
    "ko": KO_HANGUL_KEY_LINE,
    "zh": ZH_PINYIN_KEY_LINE,
    "tr": TR_KEY_LINE,
    "es": ES_KEY_LINE,
    "it": IT_KEY_LINE,
    "pt": PT_KEY_LINE,
    "nl": NL_KEY_LINE,
    "pl": PL_KEY_LINE,
    "cs": CS_KEY_LINE,
    "sv": SV_KEY_LINE,
    "no": NO_KEY_LINE,
    "da": DA_KEY_LINE,
    "fi": FI_KEY_LINE,
    "vi": VI_KEY_LINE,
    "th": TH_KEY_LINE,
}


# =====================================================================
# LANGUAGE ALPHABETS
# =====================================================================

ALPHABETS: dict[str, str] = {

    # 1 English
    "en": "abcdefghijklmnopqrstuvwxyz",

    # 2 French
    "fr": "abcdefghijklmnopqrstuvwxyzàâäæçéèêëïîôœùûüÿ",

    # 3 German
    "de": "abcdefghijklmnopqrstuvwxyzäöüß",

    # 4 Spanish
    "es": "abcdefghijklmnopqrstuvwxyzáéíóúüñ",

    # 5 Portuguese
    "pt": "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç",

    # 6 Arabic
    "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىة",

    # 7 Chinese / Pinyin
    "zh": "abcdefghijklmnopqrstuvwxyz",

    # 8 Hindi
    "hi": (
        "अआइईउऊऋएऐओऔ"
        "कखगघचछजझटठडढण"
        "तथदधनपफबभम"
        "यरलवशषसह"
        "क्षज्ञ"
    ),

    # 9 Yoruba
    "yo": "abcdefghijklmnopqrstuvwxyzáàéèẹíìóòọúùṣń",

    # 10 Hausa
    "ha": "abcdefghijklmnopqrstuvwxyzɓɗƙƴ",

    # 11 Igbo
    "ig": "abcdefghijklmnopqrstuvwxyzịñọụ",

    # 12 Swahili
    "sw": "abcdefghijklmnopqrstuvwxyz",

    # 13 Turkish
    "tr": "abcçdefgğhıijklmnoöprsştuüvyz",

    # 14 Indonesian / Malay
    "id": "abcdefghijklmnopqrstuvwxyz",

    # 15 Italian
    "it": "abcdefghijklmnopqrstuvwxyzàèéìíîòóùú",

    # Additional language alphabets
    "he": "אבגדהוזחטיכלמנסעפצקרשתךםןףץ",

    "el": "αβγδεζηθικλμνξοπρστυφχψω",

    "ru": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",

    "uk": "абвгдеєжзиіїйклмнопрстуфхцчшщьюя",

    "bn": (
        "অআইঈউঊঋএঐওঔ"
        "কখগঘঙচছজঝঞ"
        "টঠডঢণতথদধন"
        "পফবভমযরলশষসহ"
    ),

    "ja": (
        "あいうえお"
        "かきくけこ"
        "さしすせそ"
        "たちつてと"
        "なにぬねの"
        "はひふへほ"
        "まみむめも"
        "やゆよ"
        "らりるれろ"
        "わをん"
    ),

    "ko": (
        "ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ"
        "ㅁㄴㅇㄹㅎㅗㅓㅏㅣ"
        "ㅋㅌㅊㅍㅠㅜㅡ"
    ),

    "th": (
        "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธน"
        "บปผฝพฟภมยรลวศษสหฬอฮ"
    ),

    "vi": "abcdefghijklmnopqrstuvwxyzăâđêôơư",

    "nl": "abcdefghijklmnopqrstuvwxyz",

    "pl": "aąbcćdeęfghijklłmnńoóprsśtuwyzźż",

    "cs": "aábcčdďeéfghiíjklmnoópqrřsštťuúůvwxyýzž",

    "sv": "abcdefghijklmnopqrstuvwxyzåäö",

    "no": "abcdefghijklmnopqrstuvwxyzæøå",

    "da": "abcdefghijklmnopqrstuvwxyzæøå",

    "fi": "abcdefghijklmnopqrstuvwxyzåäö",
}


# =====================================================================
# DEFAULT
# =====================================================================

ALPHABETS["default"] = "abcdefghijklmnopqrstuvwxyz"
KEY_LINES["default"] = EN_KEY_LINE


# =====================================================================
# LANGUAGE ALIASES
# =====================================================================

LANG_ALIASES = {

    "eng": "en",

    "fra": "fr",
    "fre": "fr",

    "deu": "de",
    "ger": "de",

    "spa": "es",

    "por": "pt",

    "ara": "ar",

    "zho": "zh",
    "cmn": "zh",

    "hin": "hi",

    "yor": "yo",

    "hau": "ha",

    "ibo": "ig",

    "swa": "sw",

    "tur": "tr",

    "ind": "id",
    "msa": "id",
    "ms": "id",

    "ita": "it",

    "heb": "he",

    "ell": "el",
    "gre": "el",

    "rus": "ru",

    "ukr": "uk",

    "ben": "bn",

    "jpn": "ja",

    "kor": "ko",

    "tha": "th",

    "vie": "vi",

    "nld": "nl",
    "dut": "nl",

    "pol": "pl",

    "ces": "cs",
    "cze": "cs",

    "swe": "sv",

    "nor": "no",

    "dan": "da",

    "fin": "fi",

    # Nigerian Pidgin
    "pcm": "en",
}


# =====================================================================
# GRID CONSTANTS
# =====================================================================

LETTER_GRID_R = 1
WORD_GRID_R = 26

# Full-text perturbation configuration.
# These are consumed by the full-text/GSP memory layer.
FORWARD_D = 5
BACKWARD_D = 1


# =====================================================================
# STEM PREFIXES / SUFFIXES
# =====================================================================

PREFIXES = (
    "un",
    "re",
    "pre",
    "mis",
    "dis",
    "over",
    "under",
    "out",
)

SUFFIXES = (
    "tions",
    "tion",
    "ings",
    "ing",
    "edly",
    "ed",
    "es",
    "s",
    "ly",
    "ness",
    "ment",
    "able",
    "ible",
    "ers",
    "er",
    "ors",
    "or",
)


# =====================================================================
# ACCENT / CHARACTER FOLD
# =====================================================================

_FOLD = str.maketrans({

    "à": "a",
    "á": "a",
    "â": "a",
    "ä": "a",
    "ã": "a",
    "æ": "ae",

    "ç": "c",

    "è": "e",
    "é": "e",
    "ê": "e",
    "ë": "e",

    "ì": "i",
    "í": "i",
    "î": "i",
    "ï": "i",
    "ı": "i",

    "ò": "o",
    "ó": "o",
    "ô": "o",
    "ö": "o",
    "õ": "o",
    "œ": "oe",

    "ù": "u",
    "ú": "u",
    "û": "u",
    "ü": "u",

    "ÿ": "y",
    "ñ": "n",

    "ş": "s",
    "ṣ": "s",

    "ğ": "g",

    "ß": "ss",

    "ẹ": "e",
    "ọ": "o",
    "ị": "i",
    "ụ": "u",

    "ń": "n",

    "ɓ": "b",
    "ɗ": "d",
    "ƙ": "k",
    "ƴ": "y",

    "đ": "d",

    "ơ": "o",
    "ư": "u",
    "ă": "a",
})


# =====================================================================
# GLOBAL SYMBOLS BOARD
# =====================================================================

GLOBAL_SYMBOLS_BOARD: dict[str, tuple[str, ...]] = {

    "punctuation": (
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        "¿",
        "¡",
        "'",
        '"',
        "`",
        "´",
        "’",
        "‘",
        "“",
        "”",
        "…",
        "-",
        "–",
        "—",
        "_",
    ),

    "mathematical": (
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "<",
        ">",
        "≤",
        "≥",
        "≠",
        "≈",
        "≡",
        "×",
        "÷",
        "±",
        "√",
        "∞",
        "^",
        "|",
    ),

    "programming": (
        "#",
        "@",
        "$",
        "&",
        "~",
        "^",
        "*",
        "/",
        "\\",
        "%",
        "!",
        "?",
        ":",
        ";",
        ".",
        ",",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "=",
        "_",
        "`",
    ),

    "structure": (
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "/",
        "\\",
        "|",
        ":",
        ";",
        ",",
        ".",
    ),

    "currency": (
        "$",
        "€",
        "£",
        "₦",
        "¥",
        "₹",
        "₽",
        "₩",
        "₺",
        "₴",
        "₫",
        "₵",
        "₡",
        "₱",
    ),

    "logic": (
        "&",
        "|",
        "!",
        "¬",
        "∧",
        "∨",
        "→",
        "←",
        "↔",
        "⊕",
        "⊤",
        "⊥",
    ),

    "comparison": (
        "=",
        "==",
        "===",
        "!=",
        "!==",
        "<",
        ">",
        "<=",
        ">=",
        "≤",
        "≥",
    ),

    "arrows": (
        "→",
        "←",
        "↑",
        "↓",
        "↔",
        "↕",
        "⇒",
        "⇐",
        "⇔",
        "↗",
        "↘",
        "↙",
        "↖",
    ),

    "operators": (
        "+",
        "-",
        "*",
        "/",
        "%",
        "**",
        "//",
        "++",
        "--",
        "+=",
        "-=",
        "*=",
        "/=",
    ),

    "markup": (
        "#",
        "##",
        "###",
        "*",
        "**",
        "_",
        "__",
        "`",
        "```",
        ">",
        "-",
        "+",
    ),

    "social": (
        "@",
        "#",
        "&",
    ),

    "special": (
        "©",
        "®",
        "™",
        "§",
        "¶",
        "°",
        "•",
        "·",
        "†",
        "‡",
    ),
}


# =====================================================================
# SYMBOL → CATEGORY INDEX
# =====================================================================

def _build_symbol_index() -> dict[str, list[str]]:

    index: dict[str, list[str]] = {}

    for category, symbols in GLOBAL_SYMBOLS_BOARD.items():

        for symbol in symbols:

            if symbol not in index:
                index[symbol] = []

            if category not in index[symbol]:
                index[symbol].append(category)

    return index


GLOBAL_SYMBOL_INDEX = _build_symbol_index()


# =====================================================================
# SYMBOL RECOGNITION
# =====================================================================

def recognize_global_symbols(
    text: str,
) -> list[dict[str, Any]]:

    if not text:
        return []

    found: list[dict[str, Any]] = []

    for symbol, categories in GLOBAL_SYMBOL_INDEX.items():

        if symbol in text:

            found.append({
                "symbol": symbol,
                "categories": list(categories),
            })

    return found


# =====================================================================
# LANGUAGE NORMALIZATION
# =====================================================================

def normalize_lang(
    lang: str | None,
) -> str:

    if not lang:
        return "en"

    code = (
        lang
        .strip()
        .lower()
        .replace("_", "-")
        .split("-")[0]
    )

    code = LANG_ALIASES.get(
        code,
        code,
    )

    if code in ALPHABETS and code != "default":
        return code

    return "en"


# =====================================================================
# ALPHABET
# =====================================================================

def alphabet_for(
    lang: str | None,
) -> str:

    return ALPHABETS.get(
        normalize_lang(lang),
        ALPHABETS["default"],
    )


# =====================================================================
# KEY LINE
# =====================================================================

def key_line_for(
    lang: str | None,
) -> str:

    return KEY_LINES.get(
        normalize_lang(lang),
        KEY_LINES["default"],
    )


# =====================================================================
# GRID DIMENSIONS
# =====================================================================

def grid_dims(
    lang: str | None,
) -> dict[str, int]:

    """
    Return language alphabet size and grid definitions.

    The grid dimensions are linguistic dimensions.

    Letter:
        A × 1

    Word:
        A × A
    """

    A = len(
        alphabet_for(lang)
    )

    return {
        "A": A,
        "letter": f"{A}x1",
        "word": f"{A}x{A}",
    }


# =====================================================================
# STEM TOKEN
# =====================================================================

def stem_token(
    token: str,
    lang: str = "en",
) -> str:

    lang = normalize_lang(lang)

    w = (
        token or ""
    ).lower()

    alpha = alphabet_for(lang)

    w = "".join(
        ch
        for ch in w
        if ch.isalnum() or ch in alpha
    )

    if lang in {
        "ar",
        "hi",
        "bn",
        "ja",
        "ko",
        "th",
        "he",
        "el",
        "ru",
        "uk",
    }:
        return w

    if len(w) < 4:
        return w

    for pref in sorted(
        PREFIXES,
        key=len,
        reverse=True,
    ):

        if (
            w.startswith(pref)
            and len(w) - len(pref) >= 3
        ):

            w = w[
                len(pref):
            ]

            break

    for suf in sorted(
        SUFFIXES,
        key=len,
        reverse=True,
    ):

        if (
            w.endswith(suf)
            and len(w) - len(suf) >= 3
        ):

            w = w[
                : -len(suf)
            ]

            break

    return (
        w
        or (token or "").lower()
    )


# =====================================================================
# ALPHABET INDEX
# =====================================================================

def alphabet_index(
    ch: str,
    lang: str = "en",
) -> int | None:

    lang = normalize_lang(lang)

    alpha = alphabet_for(lang)

    ch = (
        ch or ""
    ).lower()

    if ch in alpha:
        return alpha.index(ch)

    folded = ch.translate(
        _FOLD
    )

    en = ALPHABETS["en"]

    if folded in en:
        return en.index(folded)

    if (
        len(folded) > 1
        and folded[0] in en
    ):
        return en.index(
            folded[0]
        )

    return None


# =====================================================================
# LETTER INDEX
# =====================================================================

def letter_index(
    ch: str,
    lang: str = "en",
) -> int | None:

    """
    Return the letter-grid placement index.

    LETTER GRID RULE:

        raw alphabet index % R

    where:

        R = 1

    Therefore every valid alphabet character maps
    into the single-row letter placement range.

    This does NOT use c.
    This does NOT use K.
    """

    index = alphabet_index(
        ch,
        lang,
    )

    if index is None:
        return None

    return index % LETTER_GRID_R


# =====================================================================
# RAW LETTER INDEX
# =====================================================================

def raw_letter_index(
    ch: str,
    lang: str = "en",
) -> int | None:

    """
    Return the unreduced alphabet index.

    This is useful when a higher layer needs the actual
    alphabet position, particularly for c.
    """

    return alphabet_index(
        ch,
        lang,
    )


# =====================================================================
# FIRST LETTER c
# =====================================================================

def first_letter_index(
    word: str,
    lang: str = "en",
) -> int:

    """
    Return c.

    c is ALWAYS the first-letter alphabet index.

    IMPORTANT:

        c is NOT modulo A.
        c is NOT modulo R.

    It remains the first-letter index itself.
    """

    if not word:
        return 0

    index = alphabet_index(
        word[0],
        lang,
    )

    if index is None:
        return 0

    return index


# =====================================================================
# LETTER CELLS
# =====================================================================

def letter_cells(
    token: str,
    lang: str = "en",
) -> list[int]:

    """
    Return letter-grid indices.

    Letter placement uses:

        alphabet_index % 1

    No K is applied.
    """

    cells: list[int] = []

    for ch in token:

        index = letter_index(
            ch,
            lang,
        )

        if index is not None:
            cells.append(index)

    return cells


# =====================================================================
# WORD INDEX
# =====================================================================

def word_index(
    token: str,
    lang: str = "en",
) -> int:

    """
    Return the word-grid row index.

    WORD GRID RULE:

        word_index % 26

    The word index is based on stem length.

    No K is applied.
    """

    stem = stem_token(
        token,
        lang,
    )

    L = max(
        len(stem),
        1,
    )

    return L % WORD_GRID_R


# =====================================================================
# WORD CELL
# =====================================================================

def word_cell(
    token: str,
    lang: str = "en",
) -> dict[str, int]:

    """
    Word-grid representation.

    PRESERVED CORE FORMULA:

        L      = len(stem)
        uID    = L
        word_S = L

    Word row:

        L + L

    reduced by:

        R = 26

    c:

        first-letter alphabet index

    c is NOT modulo-reduced.

    K is NOT applied.
    """

    lang = normalize_lang(
        lang
    )

    stem = stem_token(
        token,
        lang,
    )

    L = max(
        len(stem),
        1,
    )

    A = len(
        alphabet_for(lang)
    )

    c = first_letter_index(
     