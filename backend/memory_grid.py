from typing import List, Dict, Any
from tokenizer import tokenize, normalize_lang

class MemoryGrid:
    def __init__(self, rows=64, cols=26):
        self.rows = rows
        self.cols = cols
        self.grid = [[[] for _ in range(cols)] for _ in range(rows)]
        self.doc_store = []  # full original strings for context

    def _cell(self, token_info: dict) -> tuple[int, int]:
        L = token_info['word']['L']
        S = token_info['word']['word_S']  # same as tokenizer
        c = token_info['word']['col']     # first letter index
        row = ((L + S - 1) % self.rows) + 1
        col = c % self.cols
        return row-1, col   # zero-indexed storage

    def add_document(self, text: str, lang: str = "en", source: str = ""):
        tokens = tokenize(text, lang)
        doc_id = len(self.doc_store)
        self.doc_store.append({"text": text, "source": source, "tokens": tokens})
        for tok in tokens:
            r, c = self._cell(tok)
            self.grid[r][c].append({
                "doc_id": doc_id,
                "original": tok['original'],
                "stem": tok['stem'],
                "word": tok['word'],
                "letter": tok['letter'],
            })
        return doc_id

    def get_tokens_at(self, row: int, col: int) -> List[Dict[str, Any]]:
        row = (row - 1) % self.rows
        col = col % self.cols
        return self.grid[row][col]

    def get_doc(self, doc_id: int) -> str:
        return self.doc_store[doc_id]['text'] if doc_id < len(self.doc_store) else ""

    def get_doc_tokens(self, doc_id: int) -> list:
        return self.doc_store[doc_id]['tokens'] if doc_id < len(self.doc_store) else []