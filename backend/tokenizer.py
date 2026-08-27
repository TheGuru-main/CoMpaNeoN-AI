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

    c = first_letter_index( stem,
        lang,
    )

    # The established word formula.
    raw_word_row = L + L

    row = (
        raw_word_row
        % WORD_GRID_R
    )

    return {
        "L": L,
        "uID": L,
        "word_S": L,

        # c is constant and not modulo-reduced.
        "c": c,

        # Word-grid column is the first-letter index.
        "col": c,

        # Word-grid row uses R = 26.
        "row": row,

        "A": A,
        "R": WORD_GRID_R,
        "lang": lang,

        "grid": f"{A}x{A}",
    }


# =====================================================================
# GSP INPUT VALUES
# =====================================================================

def gsp_inputs(
    token: str,
    lang: str = "en",
) -> dict[str, int]:

    """
    Obtain the GSP-derived values from keyboard.py.

    This function does not calculate the GSP formula itself.

    keyboard.py owns:

        Lsum
        Ssum
        first-letter keyboard column
        start_row formula
    """

    lang = normalize_lang(
        lang
    )

    stem = stem_token(
        token,
        lang,
    )

    Lsum = keyboard.calculate_lsum(
        stem,
        lang,
    )

    Ssum = keyboard.calculate_ssum(
        stem,
        lang,
    )

    c = first_letter_index(
        stem,
        lang,
    )

    return {
        "Lsum": Lsum,
        "Ssum": Ssum,
        "c": c,
    }


# =====================================================================
# GSP START ROW
# =====================================================================

def gsp_start_row(
    token: str,
    lang: str = "en",
    R: int = 64,
) -> int:

    """
    Delegate start-row calculation to keyboard.py.

    Canonical formula remains:

        ((Lsum + Ssum - 1) % R) + 1
    """

    values = gsp_inputs(
        token,
        lang,
    )

    return (
        (values["Lsum"] + values["Ssum"] - 1)
        % R
    ) + 1


# =====================================================================
# TOKENIZE
# =====================================================================

def tokenize(
    text: str,
    lang: str = "en",
) -> list[dict[str, Any]]:

    lang = normalize_lang(
        lang
    )

    raw = re.sub(
        r"\s+",
        " ",
        (text or "").strip().lower(),
    )

    if not raw:
        return []

    out: list[
        dict[str, Any]
    ] = []

    for part in raw.split(" "):

        if not part:
            continue

        stem = stem_token(
            part,
            lang,
        )

        out.append({

            "original": part,

            "stem": stem,

            "lang": lang,

            "letter": letter_cells(
                stem,
                lang,
            ),

            "word": word_cell(
                stem,
                lang,
            ),

            "symbols": recognize_global_symbols(
                part
            ),

        })

    return out


# =====================================================================
# LETTER SCORE
# =====================================================================

def letter_score(
    query_tokens: list[dict],
    doc_text: str,
    lang: str = "en",
) -> float:

    lang = normalize_lang(
        lang
    )

    doc_toks = tokenize(
        doc_text,
        lang,
    )

    if (
        not query_tokens
        or not doc_toks
    ):
        return 0.0

    score = 0.0

    for qt in query_tokens:

        q_letters = (
            qt.get("letter")
            or []
        )

        if not q_letters:
            continue

        for dt in doc_toks:

            d_letters = (
                dt.get("letter")
                or []
            )

            i = 0
            j = 0
            matches = 0

            while (
                i < len(q_letters)
                and j < len(d_letters)
            ):

                if (
                    q_letters[i]
                    == d_letters[j]
                ):

                    matches += 1
                    i += 1

                j += 1

            score += (
                matches
                / max(
                    len(q_letters),
                    1,
                )
            ) * 10

    return score


# =====================================================================
# WORD SCORE
# =====================================================================

def word_score(
    query_tokens: list[dict],
    doc_text: str,
    lang: str = "en",
) -> float:

    lang = normalize_lang(
        lang
    )

    doc_toks = tokenize(
        doc_text,
        lang,
    )

    if (
        not query_tokens
        or not doc_toks
    ):
        return 0.0

    score = 0.0

    doc_cells = {
        (
            t["word"]["col"],
            t["word"]["row"],
            t["stem"],
        )
        for t in doc_toks
    }

    for qt in query_tokens:

        w = qt["word"]
        stem = qt["stem"]

        for (
            col,
            row,
            d_stem,
        ) in doc_cells:

            if stem == d_stem:

                score += 25

            elif (
                w["col"] == col
                and w["row"] == row
            ):

                score += 15

            elif (
                w["col"] == col
                or w["row"] == row
            ):

                score += 5

    return score


# =====================================================================
# SUPPORTED LANGUAGES
# =====================================================================

def supported_languages() -> list[dict[str, Any]]:

    out = []

    for code, alpha in ALPHABETS.items():

        if code == "default":
            continue

        A = len(alpha)

        out.append({

            "code": code,

            "A": A,

            "letter_grid": f"{A}x1",

            "word_grid": f"{A}x{A}",

            "key_line": KEY_LINES.get(
                code,
                KEY_LINES["default"],
            ),

        })

    return out


# =====================================================================
# LANGUAGE KEY MAPPING
# =====================================================================

def language_key_mapping(
    lang: str | None,
) -> dict[str, Any]:

    code = normalize_lang(
        lang
    )

    alpha = alphabet_for(
        code
    )

    key_line = key_line_for(
        code
    )

    return {

        "code": code,

        "alphabet": alpha,

        "key_line": key_line,

        "A": len(alpha),

        "letter_grid": f"{len(alpha)}x1",

        "word_grid": (
            f"{len(alpha)}x"
            f"{len(alpha)}"
        ),

        "letter_R": LETTER_GRID_R,

        "word_R": WORD_GRID_R,

    }


# =====================================================================
# GLOBAL SYMBOL BOARD ACCESS
# =====================================================================

def global_symbols_board(
) -> dict[str, tuple[str, ...]]:

    return {
        category: tuple(symbols)
        for category, symbols
        in GLOBAL_SYMBOLS_BOARD.items()
    }



# =====================================================================
# TEST / DEVELOPMENT
# =====================================================================

if __name__ == "__main__":

    print(
        "English mapping:"
    )

    print(
        language_key_mapping(
            "en"
        )
    )

    print(
        "\nArabic mapping:"
    )

    print(
        language_key_mapping(
            "ar"
        )
    )

    print(
        "\nChinese mapping:"
    )

    print(
        language_key_mapping(
            "zh"
        )
    )

    print(
        "\nYoruba mapping:"
    )

    print(
        language_key_mapping(
            "yo"
        )
    )

    print(
        "\nLetter index:"
    )

    print(
        alphabet_index(
            "A",
            "en",
        )
    )

    print(
        "\nLetter placement:"
    )

    print(
        letter_index(
            "A",
            "en",
        )
    )

    print(
        "\nFirst-letter c:"
    )

    print(
        first_letter_index(
            "apple",
            "en",
        )
    )

    print(
        "\nWord cell:"
    )

    print(
        word_cell(
            "deterministic",
            "en",
        )
    )

    print(
        "\nGSP inputs:"
    )

    print(
        gsp_inputs(
            "deterministic",
            "en",
        )
    )

    print(
        "\nGSP start row:"
    )

    print(
        gsp_start_row(
            "deterministic",
            "en",
        )
    )

    print(
        "\nSymbols:"
    )

    print(
        recognize_global_symbols(
            "Can GSP calculate x >= 10%?"
        )
    )

    print(
        "\nFull-text placement config:"
    )

    print(
        full_text_placement_config()
    )

    print(
        "\nSupported languages:"
    )

    for language in supported_languages():

        print(
            language
        )
     