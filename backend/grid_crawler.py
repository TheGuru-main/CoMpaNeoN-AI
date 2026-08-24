from typing import List, Set
from memory_grid import MemoryGrid

def mod_row(row, range=64):
    return ((row - 1) % range) + 1

def walk_steps(start_row, max_k=250, forward_d=5, backward_d=1):
    for k in range(max_k + 1):
        forward_row = mod_row(start_row + k * forward_d)
        backward_rows = []
        if k > 0:
            for j in range(1, 6):
                step_num = 5 * (k - 1) + j
                backward_rows.append(mod_row(start_row - step_num * backward_d))
        yield forward_row, backward_rows

def crawl(grid: MemoryGrid, start_row: int, start_col: int, limit=50):
    seen_tokens = set()
    results = []
    for frow, brows in walk_steps(start_row):
        rows = [frow] + brows
        for row in rows:
            tokens = grid.get_tokens_at(row, start_col) + grid.get_tokens_at(row, start_col)  # col from query first letter
            for tok in tokens:
                key = (tok['doc_id'], tok['original'])
                if key in seen_tokens:
                    continue
                seen_tokens.add(key)
                results.append(tok)
                if len(results) >= limit:
                    return results
    return results