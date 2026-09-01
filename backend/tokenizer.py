"""
CoMpaNeoN Tokenizer
===================

Multilingual tokenizer and alphabet-grid foundation.

Responsibilities:
- Language normalization.
- Multilingual alphabet definitions.
- Multilingual keyboard/input key lines.
- Letter-grid indexing (A × 1).
- Word-grid indexing (row % 26; col = c).
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

GSP keyboard calculations are delegated to keyboard.py when present.

GRID RULES
----------
Letter Grid:
    Shape: A × 1  (A = len(alphabet), height R = 1 → row always 0)
    Column = alphabet index (NOT index % 1)

Word Grid:
    Linguistic metadata: A × A
    Row reduction: R = 26
    row = (L + L) % 26
    col = c

FIRST LETTER c
--------------
    c = first_letter_index

c is CONSTANT for the token.
It is NOT:
    c % A
    c % 26
    derived from word-row math

K
-
K is NOT applied here.
K belongs to full-text placement and MemoryGrid.

GSP START ROW
-------------
Canonical formula (delegated / mirrored):

    start_row = ((Lsum + Ssum - 1) % R) + 1

PERTURBATION
------------
Full-text perturbation belongs to the GSP/MemoryGrid layer.

    forward_d  = 5
    backward_d = 1

The tokenizer does not apply these perturbations.
"""

from __future__ import annotations

import re
from typing import Any

try:
    import keyboard  # Companion GSP keyboard module (optional)
except ImportError:
    keyboard = None


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
PT_KEY_LINE = "QWERTYUIOP´ASDFGHJKLÇ\~ZXCVBNM"
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
        "कखगघचछजझटठडढण"
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
    "pcm": "en",
}


# =====================================================================
# GRID CONSTANTS
# =====================================================================

LETTER_GRID_R = 1   # height of letter grid (A × 1); row is always 0
WORD_GRID_R = 26    # word-row reduction for every language

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
    "à": "a", "á": "a", "â": "a", "ä": "a", "ã": "a", "æ": "ae",
    "ç": "c",
    "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ı": "i",
    "ò": "o", "ó": "o", "ô": "o", "ö": "o", "õ": "o", "œ": "oe",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "ÿ": "y", "ñ": "n",
    "ş": "s", "ṣ": "s",
    "ğ": "g",
    "ß": "ss",
    "ẹ": "e", "ọ": "o", "ị": "i", "ụ": "u",
    "ń": "n",
    "ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y",
    "đ": "d",
    "ơ": "o", "ư": "u", "ă": "a",
})


# =====================================================================
# GLOBAL SYMBOLS BOARD
# =====================================================================

GLOBAL_SYMBOLS_BOARD: dict[str, tuple[str, ...]] = {
    "punctuation": (
        ".", ",", ";", ":", "!", "?", "¿", "¡",
        "'", '"', "`", "´", "’", "‘", "“", "”",
        "…", "-", "–", "—", "_",
    ),
    "mathematical": (
        "+", "-", "*", "/", "%", "=", "<", ">",
        "≤", "≥", "≠", "≈", "≡", "×", "÷", "±", "√", "∞", "^", "|",
    ),
    "programming": (
        "#", "@", "$", "&", "\~", "^", "*", "/", "\\", "%",
        "!", "?", ":", ";", ".", ",",
        "(", ")", "[", "]", "{", "}", "<", ">", "=", "_", "`",
    ),
    "structure": (
        "(", ")", "[", "]", "{", "}", "<", ">",
        "/", "\\", "|", ":", ";", ",", ".",
    ),
    "currency": (
        "$", "€", "£", "₦", "¥", "₹", "₽", "₩", "₺", "₴", "₫", "₵", "₡", "₱",
    ),
    "logic": (
        "&", "|", "!", "¬", "∧", "∨", "→", "←", "↔", "⊕", "⊤", "⊥",
    ),
    "comparison": (
        "=", "==", "===", "!=", "!==",
        "<", ">", "<=", ">=", "≤", "≥",
    ),
    "arrows": (
        "→", "←", "↑", "↓", "↔", "↕",
        "⇒", "⇐", "⇔", "↗", "↘", "↙", "↖",
    ),
    "operators": (
        "+", "-", "*", "/", "%", "**", "//",
        "++", "--", "+=", "-=", "*=", "/=",
    ),
    "markup": (
        "#", "##", "###", "*", "**", "_", "__",
        "`", "```", ">", "-", "+",
    ),
    "social": ("@", "#", "&"),
    "special": (
        "©", "®", "™", "§", "¶", "°", "•", "·", "†", "‡",
    ),
}


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


def recognize_global_symbols(text: str) -> list[dict[str, Any]]:
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


def global_symbols_board() -> dict[str, tuple[str, ...]]:
    return {
        category: tuple(symbols)
        for category, symbols in GLOBAL_SYMBOLS_BOARD.items()
    }


def full_text_placement_config() -> dict[str, Any]:
    """GSP / MemoryGrid only — tokenizer does not apply these."""
    return {
        "forward_d": FORWARD_D,
        "backward_d": BACKWARD_D,
        "owner": "gsp_memory_grid",
        "tokenizer_applies": False,
    }


# =====================================================================
# LANGUAGE NORMALIZATION
# =====================================================================

def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    code = (
        lang.strip()
        .lower()
        .replace("_", "-")
        .split("-")[0]
    )
    code = LANG_ALIASES.get(code, code)
    if code in ALPHABETS and code != "default":
        return code
    return "en"


def alphabet_for(lang: str | None) -> str:
    return ALPHABETS.get(normalize_lang(lang), ALPHABETS["default"])


def key_line_for(lang: str | None) -> str:
    return KEY_LINES.get(normalize_lang(lang), KEY_LINES["default"])


def grid_dims(lang: str | None) -> dict[str, Any]:
    """
    Linguistic dimensions:
        letter: A × 1
        word:   A × A  (metadata)
    Operational word row uses R = 26 for all languages.
    """
    A = len(alphabet_for(lang))
    return {
        "A": A,
        "letter": f"{A}x1",
        "word": f"{A}x{A}",
        "letter_R": LETTER_GRID_R,
        "word_R": WORD_GRID_R,
    }


# =====================================================================
# STEM TOKEN
# =====================================================================

def stem_token(token: str, lang: str = "en") -> str:
    lang = normalize_lang(lang)
    w = (token or "").lower()
    alpha = alphabet_for(lang)
    w = "".join(ch for ch in w if ch.isalnum() or ch in alpha)

    if lang in {
        "ar", "hi", "bn", "ja", "ko", "th",
        "he", "el", "ru", "uk",
    }:
        return w

    if len(w) < 4:
        return w

    for pref in sorted(PREFIXES, key=len, reverse=True):
        if w.startswith(pref) and len(w) - len(pref) >= 3:
            w = w[len(pref):]
            break

    for suf in sorted(SUFFIXES, key=len, reverse=True):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break

    return w or (token or "").lower()


# =====================================================================
# ALPHABET / LETTER INDEX
# =====================================================================

def alphabet_index(ch: str, lang: str = "en") -> int | None:
    """Unreduced alphabet position (fold bridge to English when needed)."""
    lang = normalize_lang(lang)
    alpha = alphabet_for(lang)
    ch = (ch or "").lower()

    if ch in alpha:
        return alpha.index(ch)

    folded = ch.translate(_FOLD)
    en = ALPHABETS["en"]
    if folded in en:
        return en.index(folded)
    if len(folded) > 1 and folded[0] in en:
        return en.index(folded[0])
    return None


def raw_letter_index(ch: str, lang: str = "en") -> int | None:
    """Alias: unreduced alphabet index (useful for c)."""
    return alphabet_index(ch, lang)


def letter_index(ch: str, lang: str = "en") -> int | None:
    """
    Letter-grid COLUMN on A × 1.

    R = 1 means single row (row = 0), NOT (index % 1).
    Does not compute c. Does not apply K.
    """
    index = alphabet_index(ch, lang)
    if index is None:
        return None
    return index


def letter_cell(ch: str, lang: str = "en") -> dict[str, int] | None:
    """Explicit A×1 cell."""
    col = letter_index(ch, lang)
    if col is None:
        return None
    return {
        "col": col,
        "row": 0,
        "R": LETTER_GRID_R,
    }


def letter_cells(token: str, lang: str = "en") -> list[int]:
    """Sequence of letter-grid columns for lex letter scoring."""
    cells: list[int] = []
    for ch in token:
        index = letter_index(ch, lang)
        if index is not None:
            cells.append(index)
    return cells


def first_letter_index(word: str, lang: str = "en") -> int:
    """
    c = first-letter alphabet index.

    CONSTANT for the token.
    NOT modulo A.
    NOT modulo 26.
    Independent of word-row math.
    """
    if not word:
        return 0
    index = alphabet_index(word[0], lang)
    return 0 if index is None else index


# =====================================================================
# WORD INDEX / WORD CELL
# =====================================================================

def word_index(token: str, lang: str = "en") -> int:
    """
    Word-grid ROW only:
        L % 26
    """
    stem = stem_token(token, lang)
    L = max(len(stem), 1)
    return L % WORD_GRID_R


def word_cell(token: str, lang: str = "en") -> dict[str, Any]:
    """
    Word-grid representation.

    L      = len(stem)
    uID    = L
    word_S = L
    c      = first_letter_index  (NOT reduced)
    col    = c
    row    = (L + L) % 26

    No K.
    """
    lang = normalize_lang(lang)
    stem = stem_token(token, lang)
    L = max(len(stem), 1)
    A = len(alphabet_for(lang))
    c = first_letter_index(stem, lang)
    row = (L + L) % WORD_GRID_R

    return {
        "L": L,
        "uID": L,
        "word_S": L,
        "c": c,
        "col": c,
        "row": row,
        "A": A,
        "R": WORD_GRID_R,
        "lang": lang,
        "grid": f"{A}x{A}",
        "row_rule": f"(L+L)%{WORD_GRID_R}",
    }


# =====================================================================
# GSP INPUTS / START ROW (delegate only)
# =====================================================================

def _local_lsum(stem: str) -> int:
    return max(len(stem), 1)


def _local_ssum(stem: str) -> int:
    total = 0
    for ch in stem:
        if ch.isdigit():
            total += int(ch)
        else:
            total += ord(ch) % 10
    return total or 1


def gsp_inputs(token: str, lang: str = "en") -> dict[str, int]:
    """
    GSP-facing values. Tokenizer does not own storage placement.
    Prefer keyboard.calculate_lsum / calculate_ssum when present.
    """
    lang = normalize_lang(lang)
    stem = stem_token(token, lang)
    c = first_letter_index(stem, lang)

    if keyboard is not None and hasattr(keyboard, "calculate_lsum"):
        Lsum = keyboard.calculate_lsum(stem, lang)
        Ssum = keyboard.calculate_ssum(stem, lang)
    else:
        Lsum = _local_lsum(stem)
        Ssum = _local_ssum(stem)

    return {
        "Lsum": int(Lsum),
        "Ssum": int(Ssum),
        "c": int(c),
    }


def gsp_start_row(token: str, lang: str = "en", R: int = 64) -> int:
    """
    start_row = ((Lsum + Ssum - 1) % R) + 1
    """
    values = gsp_inputs(token, lang)
    if keyboard is not None and hasattr(keyboard, "start_row"):
        return int(keyboard.start_row(values["Lsum"], values["Ssum"], R))
    return ((values["Lsum"] + values["Ssum"] - 1) % R) + 1


# =====================================================================
# TOKENIZE
# =====================================================================

def tokenize(text: str, lang: str = "en") -> list[dict[str, Any]]:
    lang = normalize_lang(lang)
    raw = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not raw:
        return []

    out: list[dict[str, Any]] = []
    for part in raw.split(" "):
        if not part:
            continue
        stem = stem_token(part, lang)
        out.append({
            "original": part,
            "stem": stem,
            "lang": lang,
            "letter": letter_cells(stem, lang),
            "word": word_cell(stem, lang),
            "symbols": recognize_global_symbols(part),
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
    lang = normalize_lang(lang)
    doc_toks = tokenize(doc_text, lang)
    if not query_tokens or not doc_toks:
        return 0.0

    score = 0.0
    for qt in query_tokens:
        q_letters = qt.get("letter") or []
        if not q_letters:
            continue
        for dt in doc_toks:
            d_letters = dt.get("letter") or []
            i = j = matches = 0
            while i < len(q_letters) and j < len(d_letters):
                if q_letters[i] == d_letters[j]:
                    matches += 1
                    i += 1
                j += 1
            score += (matches / max(len(q_letters), 1)) * 10
    return score



# =====================================================================
# WORD SCORE
# =====================================================================

def word_score(
    query_tokens: list[dict],
    doc_text: str,
    lang: str = "en",
) -> float:
    lang = normalize_lang(lang)
    doc_toks = tokenize(doc_text, lang)
    if not query_tokens or not doc_toks:
        return 0.0

    score = 0.0
    doc_cells = {
        (t["word"]["col"], t["word"]["row"], t["stem"])
        for t in doc_toks
    }

    for qt in query_tokens:
        w = qt["word"]
        stem = qt["stem"]
        for col, row, d_stem in doc_cells:
            if stem == d_stem:
                score += 25
            elif w["col"] == col and w["row"] == row:
                score += 15
            elif w["col"] == col or w["row"] == row:
                score += 5
    return score


# =====================================================================
# LANGUAGE REGISTRY HELPERS
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
            "word_R": WORD_GRID_R,
            "key_line": KEY_LINES.get(code, KEY_LINES["default"]),
        })
    return out


def language_key_mapping(lang: str | None) -> dict[str, Any]:
    code = normalize_lang(lang)
    alpha = alphabet_for(code)
    key_line = key_line_for(code)
    A = len(alpha)
    return {
        "code": code,
        "alphabet": alpha,
        "key_line": key_line,
        "A": A,
        "letter_grid": f"{A}x1",
        "word_grid": f"{A}x{A}",
        "letter_R": LETTER_GRID_R,
        "word_R": WORD_GRID_R,
    }


# =====================================================================
# TEST / DEVELOPMENT
# =====================================================================

if __name__ == "__main__":
    print("English mapping:")
    print(language_key_mapping("en"))

    print("\nArabic mapping:")
    print(language_key_mapping("ar"))

    print("\nYoruba mapping:")
    print(language_key_mapping("yo"))

    print("\nLetter index A:")
    print(letter_index("A", "en"))

    print("\nLetter cell A:")
    print(letter_cell("A", "en"))

    print("\nFirst-letter c (apple):")
    print(first_letter_index("apple", "en"))

    print("\nWord cell (deterministic):")
    print(word_cell("deterministic", "en"))

    print("\nGSP inputs:")
    print(gsp_inputs("deterministic", "en"))

    print("\nGSP start row:")
    print(gsp_start_row("deterministic", "en"))

    print("\nSymbols:")
    print(recognize_global_symbols("Can GSP calculate x >= 10%?"))

    print("\nFull-text placement config:")
    print(full_text_placement_config())

    print("\nTokenize sample:")
    print(tokenize("jollof rice ₦500", "en"))