"""
Word Understanding Module

Combines:
- Letter and word tokenizer
- Memory grid retrieval
- Directive detection
- Symbol recognition
- Code term enrichment
- Candidate ranking for best context
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
from ranking import score_candidate
from intent_analyzer import detect_domain


class WordUnderstanding:
    def __init__(self, memory_grid: MemoryGrid):
        self.memory = memory_grid
        self.page_cache = PageCache()
        self.memory_cache = MemoryCache()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_context(self, query: str, lang: str = "en", limit: int = 10) -> str:
        """Retrieve and rank context from memory grid for a query."""
        cached = self.memory_cache.get(query)
        if cached:
            return cached

        tokens = tokenize(query, lang)
        if not tokens:
            return ""

        directive = detect_directive(query)
        crawl_limit = limit + 2 if directive in {"entity_identity", "location"} else limit

        first = tokens[0]
        L = first["word"]["L"]
        S = first["word"]["word_S"]
        c = first["word"]["col"]
        start_row = ((L + S - 1) % 64) + 1
        start_col = c % 26

        results = grid_crawl(self.memory, start_row, start_col, limit=crawl_limit)

        # Collect candidate documents with their text
        candidate_docs = []
        seen_doc_ids = set()
        for r in results:
            doc_id = r.get("doc_id")
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            doc_text = self.memory.get_doc(doc_id)
            if doc_text:
                candidate_docs.append({"doc_id": doc_id, "text": doc_text})

        # Rank candidates using the scoring rubric
        domain = detect_domain(query)
        ranked = []
        for doc in candidate_docs:
            score_result = score_candidate(
                query,
                doc["text"],
                query_entities=[],
                query_hierarchy=domain,
                candidate_entities=[],
                candidate_hierarchy=domain,
                freshness_score=0.0,
            )
            ranked.append((score_result["total"], doc["text"]))

        ranked.sort(key=lambda x: x[0], reverse=True)
        context = "\n".join([text for _, text in ranked[:3]])

        self.memory_cache.set(query, context)
        return context

    def score_candidate(self, query: str, doc_text: str, lang: str = "en") -> float:
        """Return combined lexical score for a candidate document."""
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