"""
Word Understanding Module

Combines:
- Letter-to-letter matching (A×1 grid)
- Word-to-word matching (A×A grid)
- Memory grid context retrieval (26×64)
- Dictionary and symbol enrichment
"""

from typing import List, Dict, Any, Optional
from tokenizer import tokenize, normalize_lang, letter_score, word_score, supported_languages
from memory_grid import MemoryGrid
from grid_crawler import crawl as grid_crawl
from symbols import recognize_symbols
from code_languages import CODE_TERMS


class WordUnderstanding:
    def __init__(self, memory_grid: MemoryGrid):
        self.memory = memory_grid

    def get_context(self, query: str, lang: str = "en", limit: int = 10) -> str:
        """
        Retrieve relevant context from the memory grid using grid crawler.
        """
        tokens = tokenize(query, lang)
        if not tokens:
            return ""

        # Use first token for starting cell
        first = tokens[0]
        L = first["word"]["L"]
        S = first["word"]["word_S"]
        c = first["word"]["col"]
        start_row = ((L + S - 1) % 64) + 1
        start_col = c % 26

        results = grid_crawl(self.memory, start_row, start_col, limit=limit)
        context_parts = []
        for r in results:
            doc_id = r["doc_id"]
            doc_text = self.memory.get_doc(doc_id)
            if doc_text:
                context_parts.append(doc_text)

        # Deduplicate and join top pieces
        unique = list(dict.fromkeys(context_parts))
        return "\n".join(unique[:3])

    def score_candidate(self, query: str, doc_text: str, lang: str = "en") -> float:
        """
        Compute a combined score for how well a document matches a query.
        Uses letter-level and word-level matching from tokenizer.
        """
        q_tokens = tokenize(query, lang)
        if not q_tokens:
            return 0.0

        # Letter score
        l_score = letter_score(q_tokens, doc_text, lang)
        # Word score
        w_score = word_score(q_tokens, doc_text, lang)

        # Symbol/code term boost
        domain_symbols = recognize_symbols(query, "general")
        if domain_symbols:
            l_score += 2.0  # small boost for symbol matches

        # Final score (weighted)
        return (l_score * 0.4) + (w_score * 0.6)

    def rank_documents(self, query: str, doc_ids: List[int], lang: str = "en") -> List[Dict[str, Any]]:
        """
        Rank documents by their score for a given query.
        """
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
        """
        High-level understanding method: returns context, ranked docs, and symbols.
        """
        context = self.get_context(query, lang)
        tokens = tokenize(query, lang)
        symbols = recognize_symbols(query, "general")
        # Get matching documents (simplified: use context)
        return {
            "context": context,
            "tokens": tokens,
            "symbols": symbols,
            "language": normalize_lang(lang),
        }