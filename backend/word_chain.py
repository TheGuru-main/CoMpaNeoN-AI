"""
Word Chain Module
=================

CoMpaNeoN relationship, continuation, and retrieval-construction layer.

Architecture:

    MemoryGrid
        ↓
    GridCrawler
        ↓
    Ranking
        ↓
    WordChain
        ├── tokenizer.py
        ├── relation_and_alphabet_matrix.py
        ├── matrix_maths.py
        └── ranking.py
        ↓
    word_understanding.py
        ↓
    AI model

Responsibilities:
- Retrieve knowledge from MemoryGrid through GridCrawler.
- Use ranking.py to rank retrieved knowledge.
- Use tokenizer.py for language-aware tokenization.
- Build adjacent word relationships.
- Build next-word transitions.
- Build phrase continuity.
- Build word-pair relationships.
- Consume the alphabet relationship matrix.
- Consume matrix mathematical weighting.
- Preserve personal/workspace/project/external source separation.
- Support multilingual knowledge.
- Provide deterministic retrieval/construction signals.

WordChain does NOT:
- Generate the final AI response.
- Replace MemoryGrid.
- Replace GridCrawler.
- Replace Ranking.
- Replace word_understanding.py.
- Replace the AI model.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

from langdetect import detect, LangDetectException

from tokenizer import (
    tokenize,
    normalize_lang,
)

from relation_and_alphabet_matrix import (
    get_word_relationships,
    get_word_relationship_classes,
    relationship_signature,
    relationship_score,
)

from matrix_maths import *

from ranking import score_candidate

from memory_grid import MemoryGrid


# ---------------------------------------------------------------------------
# MEMORY GRID
# ---------------------------------------------------------------------------

memory_grid = MemoryGrid()


# ---------------------------------------------------------------------------
# TOKEN / LANGUAGE HELPERS
# ---------------------------------------------------------------------------

WORD_PATTERN = re.compile(
    r"[a-zA-Z0-9_]+(?:[-'][a-zA-Z0-9_]+)*"
)


def detect_lang(text: str) -> str:
    """
    Detect the language of a text chunk.

    langdetect is intentionally used here because WordChain may receive
    mixed-language material from MemoryGrid.
    """

    if not text or not str(text).strip():
        return "en"

    try:
        detected = detect(str(text))
        return normalize_lang(detected)

    except LangDetectException:
        return "en"

    except Exception:
        return "en"


def clean_words(text: str) -> List[str]:
    """
    Lightweight compatibility word extraction.

    tokenizer.py remains authoritative for linguistic tokenization.
    """

    if not text:
        return []

    return WORD_PATTERN.findall(
        str(text).lower()
    )


def tokenize_text(
    text: str,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Tokenize text through the project's tokenizer.

    Language is detected dynamically when not supplied.
    """

    if not text:
        return []

    lang = normalize_lang(
        language or detect_lang(text)
    )

    try:
        return tokenize(
            str(text),
            lang,
        )

    except Exception:
        return []


def linguistic_words(
    text: str,
    language: Optional[str] = None,
) -> List[str]:
    """
    Return normalized linguistic units from tokenizer.py.

    Stems are preferred, with original tokens retained as fallback.
    """

    tokens = tokenize_text(
        text,
        language,
    )

    words = []

    for token in tokens:

        stem = token.get("stem")
        original = token.get("original")

        word = stem or original

        if word:
            words.append(
                str(word).strip().lower()
            )

    return words


# ---------------------------------------------------------------------------
# WORD CHAIN
# ---------------------------------------------------------------------------

class WordChain:

    def __init__(
        self,
        memory: Optional[MemoryGrid] = None,
    ):
        self.memory = memory or memory_grid

        # ---------------------------------------------------------------
        # GLOBAL WORD COUNTS
        # ---------------------------------------------------------------

        self.word_frequency: Counter = Counter()

        # ---------------------------------------------------------------
        # ADJACENT TRANSITIONS
        # ---------------------------------------------------------------

        self.transitions: Dict[
            str,
            Counter
        ] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # WORD PAIRS
        # ---------------------------------------------------------------

        self.pairs: Counter = Counter()

        # ---------------------------------------------------------------
        # PHRASE CHAINS
        # ---------------------------------------------------------------

        self.phrase_chains: Dict[
            str,
            Counter
        ] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # SOURCE STATISTICS
        # ---------------------------------------------------------------

        self.source_statistics: Dict[
            str,
            Counter
        ] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # LANGUAGE STATISTICS
        # ---------------------------------------------------------------

        self.language_statistics: Dict[
            str,
            Counter
        ] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # SOURCE-SPECIFIC CHAINS
        # ---------------------------------------------------------------

        self.source_chains: Dict[
            str,
            Dict[str, Any]
        ] = defaultdict(
            lambda: {
                "words": Counter(),
                "pairs": Counter(),
                "transitions": defaultdict(Counter),
                "phrases": defaultdict(Counter),
                "relationship_classes": Counter(),
                "languages": Counter(),
            }
        )

    # -------------------------------------------------------------------
    # SOURCE NORMALIZATION
    # -------------------------------------------------------------------

    @staticmethod
    def normalize_source(
        source: str,
    ) -> str:

        if not source:
            return "conversation"

        source = str(
            source
        ).strip().lower()

        allowed = {
            "personal",
            "workspace",
            "project",
            "conversation",
            "external",
            "dictionary",
            "ebook",
            "library",
            "crawler",
            "webcrawler",
            "gridcrawler",
            "ai_output",
            "ai_input",
        }

        return (
            source
            if source in allowed
            else "conversation"
        )

    # -------------------------------------------------------------------
    # ADD TEXT
    # -------------------------------------------------------------------

    def add_text(
        self,
        text: str,
        source: str = "conversation",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not text:
            return {
                "words_added": 0,
                "pairs_added": 0,
                "source": self.normalize_source(
                    source
                ),
            }

        source = self.normalize_source(
            source
        )

        lang = normalize_lang(
            language or detect_lang(text)
        )

        # ---------------------------------------------------------------
        # TOKENIZER IS THE PRIMARY LINGUISTIC INPUT
        # ---------------------------------------------------------------

        words = linguistic_words(
            text,
            lang,
        )

        # Compatibility fallback.
        if not words:
            words = clean_words(text)

        if not words:
            return {
                "words_added": 0,
                "pairs_added": 0,
                "source": source,
                "language": lang,
            }

        # ---------------------------------------------------------------
        # WORD FREQUENCY
        # ---------------------------------------------------------------

        for word in words:

            self.word_frequency[word] += 1

            self.source_chains[
                source
            ]["words"][word] += 1

        # ---------------------------------------------------------------
        # ALPHABET MATRIX
        # ---------------------------------------------------------------

        relationship_classes = []

        for word in words:

            try:

                classes = (
                    get_word_relationship_classes(
                        word
                    )
                )

                relationship_classes.extend(
                    classes
                )

                for class_id in classes:

                    self.source_chains[
                        source
                    ]["relationship_classes"][
                        str(class_id)
                    ] += 1

            except Exception:
                continue

        # ---------------------------------------------------------------
        # LANGUAGE STATISTICS
        # ---------------------------------------------------------------

        self.language_statistics[
            lang
        ]["words"] += len(words)

        self.source_chains[
            source
        ]["languages"][lang] += 1

        # ---------------------------------------------------------------
        # ADJACENT WORD RELATIONSHIPS
        # ---------------------------------------------------------------

        for i in range(
            len(words) - 1
        ):

            current_word = words[i]
            next_word = words[i + 1]

            pair = (
                f"{current_word}_{next_word}"
            )

            self.pairs[pair] += 1

            self.source_chains[
                source
            ]["pairs"][pair] += 1

            self.transitions[
                current_word
            ][next_word] += 1

            self.source_chains[
                source
            ]["transitions"][
                current_word
            ][next_word] += 1

        # ---------------------------------------------------------------
        # THREE-WORD PHRASE CHAINS
        # ---------------------------------------------------------------

        for i in range(
            len(words) - 2
        ):

            phrase = (
                f"{words[i]}_{words[i + 1]}"
            )

            next_word = words[i + 2]

            self.phrase_chains[
                phrase
            ][next_word] += 1

            self.source_chains[
                source
            ]["phrases"][
                phrase
            ][next_word] += 1

        # ---------------------------------------------------------------
        # SOURCE STATISTICS
        # ---------------------------------------------------------------

        self.source_statistics[
            source
        ]["documents"] += 1

        self.source_statistics[
            source
        ]["words"] += len(words)

        self.source_statistics[
            source
        ]["pairs"] += max(
            0,
            len(words) - 1,
        )

        return {
            "words_added": len(words),
            "pairs_added": max(
                0,
                len(words) - 1,
            ),
            "source": source,
            "language": lang,
            "relationship_classes": sorted(
                set(relationship_classes)
            ),
        }

    # -------------------------------------------------------------------
    # MEMORY GRID INGESTION
    # -------------------------------------------------------------------

    def add_memory(
        self,
        memory_record: Any,
        source: str = "personal",
    ) -> Dict[str, Any]:
        """
        Consume a memory record retrieved from MemoryGrid.

        Supports dictionaries and plain text records.
        """

        if memory_record is None:
            return {
                "words_added": 0,
                "source": source,
            }

        if isinstance(
            memory_record,
            str,
        ):

            return self.add_text(
                memory_record,
                source=source,
            )

        if isinstance(
            memory_record,
            dict,
        ):

            content = (
                memory_record.get("content")
                or memory_record.get("text")
                or memory_record.get("data")
                or ""
            )

            record_source = (
                memory_record.get("source")
                or source
            )

            language = (
                memory_record.get("language")
            )

            return self.add_text(
                str(content),
                source=record_source,
                language=language,
                metadata=memory_record,
            )

        return {
            "words_added": 0,
            "source": source,
        }

    # -------------------------------------------------------------------
    # MEMORY GRID BATCH INGESTION
    # -------------------------------------------------------------------

    def add_memory_records(
        self,
        records: Iterable[Any],
        source: str = "personal",
    ) -> Dict[str, Any]:

        documents = 0
        words = 0
        pairs = 0

        for record in records:

            result = self.add_memory(
                record,
                source=source,
            )

            documents += 1
            words += result.get(
                "words_added",
                0,
            )
            pairs += result.get(
                "pairs_added",
                0,
            )

        return {
            "documents_added": documents,
            "words_added": words,
            "pairs_added": pairs,
            "source": self.normalize_source(
                source
            ),
        }

    # -------------------------------------------------------------------
    # RANK MEMORY
    # -------------------------------------------------------------------

    def rank_memory(
        self,
        query: str,
        candidates: Iterable[Any],
        query_entities: Optional[
            List[str]
        ] = None,
        query_hierarchy: Optional[
            str
        ] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank MemoryGrid candidates through ranking.py.

        WordChain does not replace ranking.

        Ranking determines relevance first.
        WordChain then constructs relationships from the strongest
        retrieved knowledge.
        """

        ranked = []

        lang = normalize_lang(
            language or detect_lang(query)
        )

        for candidate in candidates:

            if isinstance(
                candidate,
                str,
            ):

                candidate_text = candidate
                metadata = {}

            elif isinstance(
                candidate,
                dict,
            ):

                candidate_text = (
                    candidate.get("content")
                    or candidate.get("text")
                    or candidate.get("data")
                    or ""
                )

                metadata = candidate

            else:
                continue

            if not candidate_text:
                continue

            result = score_candidate(
                query=query,
                candidate_text=str(
                    candidate_text
                ),
                candidate_entities=metadata.get(
                    "entities"
                ),
                candidate_hierarchy=metadata.get(
                    "hierarchy"
                ),
                query_entities=query_entities,
                query_hierarchy=query_hierarchy,
                freshness_score=metadata.get(
                    "freshness_score",
                    0.0,
                ),
                lang=lang,
            )

            ranked.append({
                "content": candidate_text,
                "metadata": metadata,
                "ranking": result,
            })

        ranked.sort(
            key=lambda item:
                item["ranking"]["total"],
            reverse=True,
        )

        return ranked

    # -------------------------------------------------------------------
    # RETRIEVE → RANK → BUILD
    # -------------------------------------------------------------------

    def build_from_ranked_memory(
        self,
        query: str,
        candidates: Iterable[Any],
        source: str = "personal",
        limit: int = 20,
        query_entities: Optional[
            List[str]
        ] = None,
        query_hierarchy: Optional[
            str
        ] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main retrieval construction pipeline.

            MemoryGrid
                 ↓
            candidates
                 ↓
             ranking
                 ↓
             WordChain
        """

        ranked = self.rank_memory(
            query=query,
            candidates=candidates,
            query_entities=query_entities,
            query_hierarchy=query_hierarchy,
            language=language,
        )

        selected = ranked[:limit]

        ingestion = []

        for item in selected:

            result = self.add_text(
                item["content"],
                source=source,
                metadata=item["metadata"],
                language=language,
            )

            ingestion.append({
                "ranking": item["ranking"],
                "ingestion": result,
            })

        return {
            "query": query,
            "ranked": selected,
            "ingestion": ingestion,
            "count": len(selected),
        }

    # -------------------------------------------------------------------
    # NEXT WORD
    # -------------------------------------------------------------------

    def next_words(
        self,
        word: str,
        limit: int = 5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not word:
            return []

        word = str(
            word
        ).lower().strip()

        if source:

            source = self.normalize_source(
                source
            )

            counter = (
                self.source_chains[
                    source
                ]["transitions"].get(
                    word,
                    Counter(),
                )
            )

        else:

            counter = self.transitions.get(
                word,
                Counter(),
            )

        results = []

        total = sum(
            counter.values()
        )

        for next_word, count in (
            counter.most_common(limit)
        ):

            results.append({
                "word": next_word,
                "after": word,
                "count": count,
                "probability": self._probability(
                    count,
                    total,
                ),
                "source": source,
            })

        return result