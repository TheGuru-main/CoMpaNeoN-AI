"""
Letter grid A×1 + word grid A×A.
Stem prefixes/affixes. Alphabets aligned with GSP keyboard language packs (15).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 15 production language alphabets (ordered index → letter grid column)
# Keyboard maps define layout; these strings define search token indices.
# ---------------------------------------------------------------------------
ALPHABETS: dict[str, str] = {
    # 1 English — QWERTY base
    "en": "abcdefghijklmnopqrstuvwxyz",
    # 2 French — AZERTY + accents
    "fr": "abcdefghijklmnopqrstuvwxyzàâäæçéèêëïîôœùûüÿ",
    # 3 German — QWERTZ + umlauts
    "de": "abcdefghijklmnopqrstuvwxyzäöüß",
    # 4 Spanish
    "es": "abcdefghijklmnopqrstuvwxyzáéíóúüñ",
    # 5 Portuguese
    "pt": "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç",
    # 6 Arabic (standard letter set for token index)
    "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىة",
    # 7 Chinese — Pinyin Latin used by virtual keyboard map
    "zh": "abcdefghijklmnopqrstuvwxyz",
    # 8 Hindi — Devanagari core + inherent vowels subset
    "hi": "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहक्षज्ञ",
    # 9 Yoruba (Nigeria core) — Latin + common tone vowels
    "yo": "abcdefghijklmnopqrstuvwxyzáàéèẹíìóòọúùṣń",
    # 10 Hausa (Nigeria core)
    "ha": "abcdefghijklmnopqrstuvwxyzɓɗƙƴ",
    # 11 Igbo (Nigeria core)
    "ig": "abcdefghijklmnopqrstuvwxyzịñọụ",
    # 12 Swahili
    "sw": "abcdefghijklmnopqrstuvwxyz",
    # 13 Turkish
    "tr": "abcçdefgğhıijklmnoöprsştuüvyz",
    # 14 Indonesian / Malay
    "id": "abcdefghijklmnopqrstuvwxyz",
    # 15 Italian
    "it": "abcdefghijklmnopqrstuvwxyzàèéìíîòóùú",
    # fallback
    "default": "abcdefghijklmnopqrstuvwxyz",
}

# ISO-ish aliases → primary code
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
    "pcm": "en",  # Nigerian Pidgin → English letter grid + commerce dict
}

PREFIXES = ("un", "re", "pre", "mis", "dis", "over", "under", "out")
SUFFIXES = (
    "tions", "tion", "ings", "ing", "edly", "ed", "es", "s",
    "ly", "ness", "ment", "able", "ible", "ers", "er", "ors", "or",
)

# Accent fold → English base for cross-lang weak match
_FOLD = str.maketrans({
    "à": "a", "á": "a", "â": "a", "ä": "a", "ã": "a", "æ": "ae",
    "ç": "c",
    "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ı": "i",
    "ò": "o", "ó": "o", "ô": "o", "ö": "o", "õ": "o", "œ": "oe",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "ÿ": "y", "ñ": "n", "ş": "s", "ṣ": "s", "ğ": "g", "ß": "ss",
    "ẹ": "e", "ọ": "o", "ị": "i", "ụ": "u", "ń": "n",
    "ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y",
})


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    code = lang.strip().lower().replace("_", "-").split("-")[0]
    code = LANG_ALIASES.get(code, code)
    if code in ALPHABETS and code != "default":
        return code
    return "en"


def alphabet_for(lang: str | None) -> str:
    return ALPHABETS.get(normalize_lang(lang), ALPHABETS["default"])


def grid_dims(lang: str | None) -> dict[str, int]:
    """Letter grid A×1, word grid A×A."""
    A = len(alphabet_for(lang))
    return {"A": A, "letter": f"{A}x1", "word": f"{A}x{A}"}


def stem_token(token: str, lang: str = "en") -> str:
    lang = normalize_lang(lang)
    # Keep letters/marks that appear in this language alphabet + ascii alnum
    w = (token or "").lower()
    alpha = alphabet_for(lang)
    w = "".join(ch for ch in w if ch.isalnum() or ch in alpha)
    if lang in {"ar", "hi"}:
        return w  # no English affix stem on Arabic/Devanagari
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


def letter_index(ch: str, lang: str = "en") -> int | None:
    lang = normalize_lang(lang)
    alpha = alphabet_for(lang)
    ch = (ch or "").lower()
    if ch in alpha:
        return alpha.index(ch)
    # fold then try English grid (code-mix bridge)
    folded = ch.translate(_FOLD)
    en = ALPHABETS["en"]
    if folded in en:
        return en.index(folded)
    if len(folded) > 1 and folded[0] in en:
        return en.index(folded[0])
    return None


def letter_cells(token: str, lang: str = "en") -> list[int]:
    return [i for ch in token if (i := letter_index(ch, lang)) is not None]


def word_cell(token: str, lang: str = "en") -> dict[str, int]:
    """Word grid A×A: L=len, uID=L, word_S=L."""
    lang = normalize_lang(lang)
    stem = stem_token(token, lang)
    L = max(len(stem), 1)
    A = len(alphabet_for(lang))
    first = letter_index(stem[0], lang) if stem else 0
    if first is None:
        first = 0
    return {
        "L": L,
        "uID": L,
        "word_S": L,
        "col": first % A,
        "row": (L + L) % A,
        "A": A,
        "lang": lang,
        "grid": f"{A}x{A}",
    }


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
        })
    return out


def letter_score(query_tokens: list[dict], doc_text: str, lang: str = "en") -> float:
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


def word_score(query_tokens: list[dict], doc_text: str, lang: str = "en") -> float:
    lang = normalize_lang(lang)
    doc_toks = tokenize(doc_text, lang)
    if not query_tokens or not doc_toks:
        return 0.0
    score = 0.0
    doc_cells = {(t["word"]["col"], t["word"]["row"], t["stem"]) for t in doc_toks}
    for qt in query_tokens:
        w, stem = qt["word"], qt["stem"]
        for col, row, d_stem in doc_cells:
            if stem == d_stem:
                score += 25
            elif (w["col"], w["row"]) == (col, row):
                score += 15
            elif w["col"] == col or w["row"] == row:
                score += 5
    return score


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
        })
    return out
