"""
Word Understanding Module (updated)

Combines:
- Letter and word tokenizer
- Memory grid retrieval
- Directive detection
- Symbol recognition
- Code term enrichment
- Content caching for context
"""

import hashlib
from typing import List, Dict, Any, Optional
from tokenizer import tokenize, normalize_lang, letter_score, word_score, supported_languages
from memory_grid import MemoryGrid
from grid_crawler import crawl as grid_crawl
from symbols import recognize_symbols
from code_languages import CODE_TERMS
from directives import detect_directive
from page_cache import PageCache
from memory_cache import MemoryCache

class WordUnderstanding:
    def __init__(self, memory_grid: MemoryGrid):
        self.memory = memory_grid
        self.page_cache = PageCache()
        self.memory_cache = MemoryCache()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_context(self, query: str, lang: str = "en", limit: int = 10) -> str:
        # Check memory cache first (no TTL)
        cached = self.memory_cache.get(query)
        if cached:
            return cached

        tokens = tokenize(query, lang)
        if not tokens:
            return ""

        directive = detect_directive(query)
        # Adjust grid crawl based on directive (simplified: adjust limit)
        if directive in {"entity_identity", "location"}:
            crawl_limit = limit + 2
        else:
            crawl_limit = limit

        first = tokens[0]
        L = first["word"]["L"]
        S = first["word"]["word_S"]
        c = first["word"]["col"]
        start_row = ((L + S - 1) % 64) + 1
        start_col = c % 26

        results = grid_crawl(self.memory, start_row, start_col, limit=crawl_limit)
        context_parts = []
        for r in results:
            doc_id = r["doc_id"]
            doc_text = self.memory.get_doc(doc_id)
            if doc_text:
                context_parts.append(doc_text)

        context = "\n".join(dict.fromkeys(context_parts[:3]))
        self.memory_cache.set(query, context)
        return context

    def score_candidate(self, query: str, doc_text: str, lang: str = "en") -> float:
        q_tokens = tokenize(query, lang)
        if not q_tokens:
            return 0.0
        l_score = letter_score(q_tokens, doc_text, lang)
        w_score = word_score(q_tokens, doc_text, lang)
        return (l_score * 0.4) + (w_score * 0.6)

    def rank_documents(self, query: str, doc_ids: List[int], lang: str = "en") -> List[Dict[str, Any]]:
        scored = []
        for doc_id in doc_ids:
            text = self.memory.get_doc(doc_id)
            if not text:
                continue
            score = self.score_candidate(query, text, lang)
            scored.append({"doc_id": doc_id, "score": score, "text": text})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def understand(self, query: str, lang: str = "en") -> dict:
        context = self.get_context(query, lang)
        tokens = tokenize(query, lang)
        symbols = recognize_symbols(query, "general")
        directive = detect_directive(query)
        return {
            "context": context,
            "tokens": tokens,
            "symbols": symbols,
            "directive": directive,
            "language": normalize_lang(lang),
        }