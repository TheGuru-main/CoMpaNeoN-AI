"""
CoMpaNeoN Tokenizer
===================

Multilingual lexical tokenizer and deterministic identity foundation.

The tokenizer is responsible for:

- language normalization
- multilingual character/alphabet handling
- multilingual keyboard/input metadata
- Unicode-safe tokenization
- lexical normalization
- character decomposition
- 0-based alphabet indexing
- ordered lexical UID generation
- lexical UID serialization
- lexical UID S calculation
- L / S / SC generation
- 46 × 26 word-grid placement inputs
- letter-grid indexing
- symbol recognition
- lightweight stem metadata
- lexical similarity signals
- compatibility helpers used by existing CoMpaNeoN modules

The tokenizer does NOT own:

- GSP crawler traversal
- GSP K traversal
- GSP forward/backward jumps
- XOR sharding
- quorum routing
- MemoryGrid storage
- full-text placement
- full-text UID chaining
- ranking policy
- WordChain
- WordUnderstanding
- POS analysis
- linguistic analysis
- semantic analysis
- dictionary enrichment
- prompt management
- AI response generation

IMPORTANT IDENTITY RULE
-----------------------

Alphabet indexes are ZERO-BASED.

English:

    A = 0
    B = 1
    ...
    Z = 25

Example:

    ZED -> [25, 4, 3]
    ZEE -> [25, 4, 4]

The ordered UID sequence is preserved.

Serialized UID:

    ZED -> "2543"
    ZEE -> "2544"

S is NOT the digit sum of the serialized UID.

Instead:

    S = sum(uid_sequence)

Therefore:

    ZED -> 25 + 4 + 3 = 32
    ZEE -> 25 + 4 + 4 = 33

WORD GRID
---------

The word grid remains:

    46 columns × 26 rows

Word placement uses:

    row = ((L + S - 1) % 26) + 1

where:

    L = normalized lexical token length
    S = sum(uid_sequence)

SC is:

    SC = first-letter alphabet index

SC is NOT a random value and is NOT derived from word-row math.

FULL TEXT
---------

The tokenizer preserves each word's ordered UID sequence.

The full-text placement layer may then use those UID sequences for
column-index chaining and calculate the total/full-text UID and S.

The tokenizer does not perform full-text placement.

GSP START ROW
-------------

Lexical UID identity is separate from the GSP keyboard-placement
authority.

When a GSP start row is requested, this module delegates to keyboard.py
when available.

The tokenizer does not apply crawler K/D traversal.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


# ---------------------------------------------------------------------
# Optional GSP keyboard companion
# ---------------------------------------------------------------------

try:
    import keyboard  # type: ignore
except ImportError:
    keyboard = None


# =====================================================================
# MULTILINGUAL KEY LINES
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
    "en": "abcdefghijklmnopqrstuvwxyz",

    "fr": "abcdefghijklmnopqrstuvwxyzàâäæçéèêëïîôœùûüÿ",
    "de": "abcdefghijklmnopqrstuvwxyzäöüß",
    "es": "abcdefghijklmnopqrstuvwxyzáéíóúüñ",
    "pt": "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç",

    "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىة",

    "zh": "abcdefghijklmnopqrstuvwxyz",

    "hi": (
        "अआइईउऊऋएऐओऔ"
        "कखगघङचछजझञटठडढण"
        "तथदधनपफबभम"
        "यरलवशषसह"
        "क्षज्ञ"
    ),

    "yo": "abcdefghijklmnopqrstuvwxyzáàéèẹíìóòọúùṣń",
    "ha": "abcdefghijklmnopqrstuvwxyzɓɗƙƴ",
    "ig": "abcdefghijklmnopqrstuvwxyzịñọụ",
    "sw": "abcdefghijklmnopqrstuvwxyz",

    "tr": "abcçdefgğhıijklmnoöprsştuüvyz",
    "id": "abcdefghijklmnopqrstuvwxyz",
    "it": "abcdefghijklmnopqrstuvwxyzàèéìíîòóùú",

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

ALPHABETS["default"] = ALPHABETS["en"]
KEY_LINES["default"] = KEY_LINES["en"]


# =====================================================================
# LANGUAGE ALIASES
# =====================================================================

LANG_ALIASES: dict[str, str] = {
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
    "mandarin": "zh",
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
    "pcm": "en",
}


# =====================================================================
# GRID CONSTANTS
# =====================================================================

# Letter grid is one row.
LETTER_GRID_R = 1

# Word placement is always reduced against 26 rows.
WORD_GRID_R = 26

# The physical word grid is 46 columns × 26 rows.
WORD_GRID_COLUMNS = 46
WORD_GRID_ROWS = 26

# These remain metadata only.
FORWARD_D = 5
BACKWARD_D = 1


# =====================================================================
# STEM METADATA
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
# CHARACTER FOLDING
# =====================================================================

_FOLD = str.maketrans(
    {
        "à": "a",
        "á": "a",
        "â": "a",
        "ä": "a",
        "ã": "a",
        "å": "a",
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
    }
)


# =====================================================================
# GLOBAL SYMBOL BOARD
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


def _build_symbol_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}

    for category, symbols in GLOBAL_SYMBOLS_BOARD.items():
        for symbol in symbols:
            index.setdefault(symbol, [])

            if category not in index[symbol]:
                index[symbol].append(category)

    return index


GLOBAL_SYMBOL_INDEX = _build_symbol_index()


def recognize_global_symbols(text: str) -> list[dict[str, Any]]:
    """
    Return recognized symbols without altering the original text.
    """

    if not text:
        return []

    found: list[dict[str, Any]] = []

    # Longest symbols first so === is recognized before =.
    symbols = sorted(
        GLOBAL_SYMBOL_INDEX.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for symbol, categories in symbols:
        if symbol in text:
            found.append(
                {
                    "symbol": symbol,
                    "categories": list(categories),
                }
            )

    return found


def global_symbols_board() -> dict[str, tuple[str, ...]]:
    return {
        category: tuple(symbols)
        for category, symbols in GLOBAL_SYMBOLS_BOARD.items()
    }


# =====================================================================
# LANGUAGE NORMALIZATION
# =====================================================================

def normalize_lang(lang: str | None) -> str:
    """
    Normalize a language code.

    Unknown language codes are preserved as lower-case codes rather
    than forcing the text itself into English.
    """

    if not lang:
        return "en"

    code = (
        str(lang)
        .strip()
        .lower()
        .replace("_", "-")
    )

    code = code.split("-")[0]

    return LANG_ALIASES.get(code, code)


def alphabet_for(lang: str | None) -> str:
    """
    Return the known alphabet for a language.

    Unknown languages receive the generic Unicode alphabet sentinel.
    Their actual characters are indexed deterministically through the
    fallback alphabet builder.
    """

    code = normalize_lang(lang)

    if code in ALPHABETS:
        return ALPHABETS[code]

    return ALPHABETS["default"]


def key_line_for(lang: str | None) -> str:
    code = normalize_lang(lang)
    return KEY_LINES.get(code, KEY_LINES["default"])


def _unicode_alphabet_for_text(text: str) -> str:
    """
    Build a deterministic alphabet for an unsupported script/language.

    This is intentionally based on Unicode code-point ordering rather
    than an arbitrary encounter order.

    Known language alphabets always take precedence.
    """

    chars: set[str] = set()

    for ch in text:
        if ch.isalpha():
            chars.add(ch.lower())

    return "".join(sorted(chars, key=lambda c: ord(c)))


def grid_dims(lang: str | None) -> dict[str, Any]:
    """
    Report tokenizer/grid dimensions.

    Letter-grid dimensions follow the active language alphabet.

    Word placement itself remains fixed at:

        46 × 26
    """

    code = normalize_lang(lang)
    alpha = alphabet_for(code)

    return {
        "A": len(alpha),
        "letter": f"{len(alpha)}x1",
        "word": f"{WORD_GRID_COLUMNS}x{WORD_GRID_ROWS}",
        "letter_R": LETTER_GRID_R,
        "word_R": WORD_GRID_R,
        "word_columns": WORD_GRID_COLUMNS,
        "word_rows": WORD_GRID_ROWS,
        "lang": code,
    }


# =====================================================================
# NORMALIZATION
# =====================================================================

def normalize_text(text: str | None) -> str:
    """
    Unicode-safe general text normalization.

    Keeps letters, numbers, whitespace and symbols.
    """

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Normalize whitespace without destroying non-Latin scripts.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_lexical_token(
    token: str | None,
    lang: str | None = "en",
) -> str:
    """
    Normalize a lexical token while preserving its script.

    IMPORTANT:
    - punctuation around a word is not part of lexical identity
    - Unicode letters/numbers are retained
    - the lexical UID is based on this full normalized token
    - stemming does NOT alter UID identity
    """

    if token is None:
        return ""

    text = unicodedata.normalize("NFKC", str(token)).strip().lower()

    if not text:
        return ""

    # Keep every Unicode letter/number.
    chars = []

    for ch in text:
        if ch.isalnum():
            chars.append(ch)

    return "".join(chars)


# =====================================================================
# STEM METADATA
# =====================================================================

def stem_token(token: str, lang: str = "en") -> str:
    """
    Lightweight stem metadata.

    The stem is NOT used to generate lexical UID.

    This function remains intentionally conservative for non-Latin
    languages.
    """

    code = normalize_lang(lang)
    w = normalize_lexical_token(token, code)

    if not w:
        return ""

    if code in {
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

    for pref in sorted(PREFIXES, key=len, reverse=True):
        if w.startswith(pref) and len(w) - len(pref) >= 3:
            w = w[len(pref):]
            break

    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break

    return w or normalize_lexical_token(token, code)


# =====================================================================
# ALPHABET INDEXING
# =====================================================================

def alphabet_index(
    ch: str,
    lang: str = "en",
    text_context: str | None = None,
) -> int | None:
    """
    Return the deterministic ZERO-BASED alphabet index.

    For known languages:
        use the configured language alphabet.

    For unknown languages/scripts:
        construct a deterministic Unicode alphabet from the supplied
        text context.

    English example:

        A -> 0
        Z -> 25
    """

    if not ch:
        return None

    code = normalize_lang(lang)
    ch = str(ch)[0].lower()

    alpha = ALPHABETS.get(code)

    if alpha is not None:
        if ch in alpha:
            return alpha.index(ch)

        # Accent/code-mix bridge.
        folded = ch.translate(_FOLD)

        if folded in alpha:
            return alpha.index(folded)

        # Latin bridge to English.
        if folded in ALPHABETS["en"]:
            return ALPHABETS["en"].index(folded)

        if len(folded) > 1 and folded[0] in ALPHABETS["en"]:
            return ALPHABETS["en"].index(folded[0])

        return None

    # Unknown language/script.
    context = text_context or ch
    dynamic_alpha = _unicode_alphabet_for_text(context)

    if ch in dynamic_alpha:
        return dynamic_alpha.index(ch)

    return None


def raw_letter_index(
    ch: str,
    lang: str = "en",
    text_context: str | None = None,
) -> int | None:
    """
    Alias for the unreduced zero-based alphabet index.
    """

    return alphabet_index(ch, lang, text_context)


def letter_index(
    ch: str,
    lang