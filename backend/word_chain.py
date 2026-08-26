"""
Word Chain Module
=================

CoMpaNeoN word relationship and continuation layer.

Responsibilities:
- Build adjacent word relationships.
- Build next-word transitions.
- Track phrase continuity.
- Track word-pair frequency.
- Support external knowledge sources such as dictionaries, ebooks,
  libraries, and crawled text.
- Support personal/workspace knowledge separately from external knowledge.
- Provide deterministic predictions for retrieval, follow-up, analysis,
  planning, and response generation.
- Provide serializable state for persistent personal AI learning.

Important distinction:

UNDERSTANDING != RESPONSE

WordChain does not generate the final AI response.
It provides relationships that other modules can use to understand
context and construct a response.

The module is deliberately independent from:
- ai_model.py
- memory_grid.py
- grid_crawler.py
- ranking.py
- summary.py
- follow_up.py

Those modules may consume WordChain.

Knowledge sources:
    personal
    workspace
    project
    conversation
    external
    dictionary
    ebook
    library
    crawler

Personal/private data must remain distinguishable from external knowledge.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# TOKEN CLEANING
# ---------------------------------------------------------------------------

WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+(?:[-'][a-zA-Z0-9_]+)*")


def clean_words(text: str) -> List[str]:
    """
    Convert text into a deterministic lowercase word sequence.

    This is intentionally lightweight.

    tokenizer.py remains responsible for the deeper linguistic tokenization.
    WordChain works on a stable word sequence for relationships.
    """

    if not text:
        return []

    return WORD_PATTERN.findall(str(text).lower())


# ---------------------------------------------------------------------------
# WORD CHAIN
# ---------------------------------------------------------------------------

class WordChain:
    """
    Persistent word relationship engine.

    The chain stores:

        word -> next word -> count

    and:

        word pair -> frequency

    It also stores source information so personal knowledge and external
    knowledge can remain separated.
    """

    def __init__(self):
        # ---------------------------------------------------------------
        # GLOBAL WORD COUNTS
        # ---------------------------------------------------------------

        self.word_frequency: Counter = Counter()

        # ---------------------------------------------------------------
        # ADJACENT WORD RELATIONSHIPS
        #
        # transitions["python"]["language"] = 12
        # ---------------------------------------------------------------

        self.transitions: Dict[str, Counter] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # WORD PAIRS
        #
        # pairs["python_is"] = 4
        # ---------------------------------------------------------------

        self.pairs: Counter = Counter()

        # ---------------------------------------------------------------
        # TRIPLE / SHORT PHRASE CONTINUITY
        #
        # phrase_chains["machine_learning"]["model"] = ...
        #
        # Key is the preceding phrase.
        # ---------------------------------------------------------------

        self.phrase_chains: Dict[str, Counter] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # SOURCE COUNTS
        #
        # Keeps source classes separate.
        # ---------------------------------------------------------------

        self.source_statistics: Dict[str, Counter] = defaultdict(Counter)

        # ---------------------------------------------------------------
        # SOURCE-SPECIFIC CHAINS
        #
        # This allows:
        #
        # personal knowledge
        # workspace knowledge
        # project knowledge
        # external knowledge
        #
        # to be queried independently.
        # ---------------------------------------------------------------

        self.source_chains: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: {
                "words": Counter(),
                "pairs": Counter(),
                "transitions": defaultdict(Counter),
                "phrases": defaultdict(Counter),
            }
        )

    # -------------------------------------------------------------------
    # SOURCE NORMALIZATION
    # -------------------------------------------------------------------

    @staticmethod
    def normalize_source(source: str) -> str:
        """
        Normalize the knowledge source.

        Supported conceptual source classes:

            personal
            workspace
            project
            conversation
            external
            dictionary
            ebook
            library
            crawler
        """

        if not source:
            return "conversation"

        source = str(source).strip().lower()

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
        }

        return source if source in allowed else "conversation"

    # -------------------------------------------------------------------
    # ADD TEXT
    # -------------------------------------------------------------------

    def add_text(
        self,
        text: str,
        source: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add text to the chain.

        Returns a compact ingestion report.

        metadata is intentionally accepted but not stored directly here.
        Memory ownership belongs to the memory layer.

        WordChain stores linguistic relationships, not the user's entire
        memory database.
        """

        words = clean_words(text)

        if not words:
            return {
                "words_added": 0,
                "pairs_added": 0,
                "source": self.normalize_source(source),
            }

        source = self.normalize_source(source)

        # ---------------------------------------------------------------
        # WORD FREQUENCY
        # ---------------------------------------------------------------

        for word in words:
            self.word_frequency[word] += 1
            self.source_chains[source]["words"][word] += 1

        # ---------------------------------------------------------------
        # ADJACENT PAIRS / TRANSITIONS
        # ---------------------------------------------------------------

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            pair = f"{current_word}_{next_word}"

            self.pairs[pair] += 1
            self.source_chains[source]["pairs"][pair] += 1

            self.transitions[current_word][next_word] += 1
            self.source_chains[source]["transitions"][
                current_word
            ][next_word] += 1

        # ---------------------------------------------------------------
        # SHORT PHRASE CHAINS
        #
        # Example:
        #
        # "machine learning model"
        #
        # machine_learning -> model
        # ---------------------------------------------------------------

        for i in range(len(words) - 2):
            phrase = f"{words[i]}_{words[i + 1]}"
            next_word = words[i + 2]

            self.phrase_chains[phrase][next_word] += 1

            self.source_chains[source]["phrases"][
                phrase
            ][next_word] += 1

        # ---------------------------------------------------------------
        # SOURCE STATISTICS
        # ---------------------------------------------------------------

        self.source_statistics[source]["documents"] += 1
        self.source_statistics[source]["words"] += len(words)
        self.source_statistics[source]["pairs"] += max(
            0,
            len(words) - 1,
        )

        return {
            "words_added": len(words),
            "pairs_added": max(0, len(words) - 1),
            "source": source,
        }

    # -------------------------------------------------------------------
    # ADD MANY DOCUMENTS
    # -------------------------------------------------------------------

    def add_documents(
        self,
        documents: Iterable[str],
        source: str = "external",
    ) -> Dict[str, Any]:
        """
        Add multiple documents.

        Useful for:
        - dictionaries
        - ebooks
        - libraries
        - crawler results
        - workspace imports
        """

        count = 0
        words = 0
        pairs = 0

        for document in documents:
            result = self.add_text(
                document,
                source=source,
            )

            count += 1
            words += result["words_added"]
            pairs += result["pairs_added"]

        return {
            "documents_added": count,
            "words_added": words,
            "pairs_added": pairs,
            "source": self.normalize_source(source),
        }

    # -------------------------------------------------------------------
    # NEXT WORD CANDIDATES
    # -------------------------------------------------------------------

    def next_words(
        self,
        word: str,
        limit: int = 5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return likely next words.

        Results are deterministic and frequency-ranked.
        """

        if not word:
            return []

        word = str(word).lower().strip()

        if source:
            source = self.normalize_source(source)

            counter = self.source_chains[source]["transitions"].get(
                word,
                Counter(),
            )
        else:
            counter = self.transitions.get(
                word,
                Counter(),
            )

        return [
            {
                "word": next_word,
                "after": word,
                "count": count,
                "probability": self._probability(
                    count,
                    sum(counter.values()),
                ),
                "source": source,
            }
            for next_word, count in counter.most_common(limit)
        ]

    # -------------------------------------------------------------------
    # NEXT WORD FROM PHRASE
    # -------------------------------------------------------------------

    def next_words_from_phrase(
        self,
        phrase: str,
        limit: int = 5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict the continuation of a two-word phrase.
        """

        words = clean_words(phrase)

        if len(words) < 2:
            return []

        key = f"{words[-2]}_{words[-1]}"

        if source:
            source = self.normalize_source(source)

            counter = self.source_chains[source]["phrases"].get(
                key,
                Counter(),
            )
        else:
            counter = self.phrase_chains.get(
                key,
                Counter(),
            )

        return [
            {
                "word": next_word,
                "after_phrase": key,
                "count": count,
                "probability": self._probability(
                    count,
                    sum(counter.values()),
                ),
                "source": source,
            }
            for next_word, count in counter.most_common(limit)
        ]

    # -------------------------------------------------------------------
    # WORD PAIRS
    # -------------------------------------------------------------------

    def get_pairs(
        self,
        word: Optional[str] = None,
        limit: int = 20,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve learned word pairs.

        If word is provided, only pairs containing that word are returned.
        """

        if source:
            source = self.normalize_source(source)
            counter = self.source_chains[source]["pairs"]
        else:
            counter = self.pairs

        results = []

        for pair, count in counter.most_common():

            if word:
                normalized = str(word).lower()

                if normalized not in pair.split("_"):
                    continue

            parts = pair.split("_", 1)

            if len(parts) != 2:
                continue

            results.append({
                "pair": pair,
                "first": parts[0],
                "second": parts[1],
                "count": count,
                "source": source,
            })

            if len(results) >= limit:
                break

        return results

    # -------------------------------------------------------------------
    # PHRASE CONTINUITY
    # -------------------------------------------------------------------

    def phrase_continuity(
        self,
        query: str,
        candidate_text: str,
    ) -> float:
        """
        Calculate phrase continuity between two texts.

        Returns:
            0.0 - 100.0
        """

        query_words = clean_words(query)
        candidate_words = clean_words(candidate_text)

        if not query_words or not candidate_words:
            return 0.0

        longest = 0

        # Search contiguous sequences.
        for start in range(len(query_words)):

            for candidate_start in range(
                len(candidate_words)
            ):

                current = 0

                while (
                    start + current < len(query_words)
                    and candidate_start + current
                    < len(candidate_words)
                    and query_words[start + current]
                    == candidate_words[
                        candidate_start + current
                    ]
                ):
                    current += 1

                longest = max(
                    longest,
                    current,
                )

        return (
            longest / len(query_words)
        ) * 100.0

    # -------------------------------------------------------------------
    # WORD-PAIR MATCH
    # -------------------------------------------------------------------

    def pair_overlap(
        self,
        query: str,
        candidate_text: str,
    ) -> float:
        """
        Measure overlapping adjacent word pairs.

        Returns:
            0.0 - 100.0
        """

        query_words = clean_words(query)
        candidate_words = clean_words(candidate_text)

        if len(query_words) < 2:
            return 0.0

        query_pairs = {
            f"{query_words[i]}_{query_words[i + 1]}"
            for i in range(len(query_words) - 1)
        }

        candidate_pairs = {
            f"{candidate_words[i]}_{candidate_words[i + 1]}"
            for i in range(len(candidate_words) - 1)
        }

        if not query_pairs:
            return 0.0

        matched = len(
            query_pairs & candidate_pairs
        )

        return (
            matched / len(query_pairs)
        ) * 100.0

    # -------------------------------------------------------------------
    # CONTEXT CONTINUATION
    # -------------------------------------------------------------------

    def predict_from_text(
        self,
        text: str,
        limit: int = 5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict continuation from the final word(s) of a text.

        Two-word context is preferred.
        """

        words = clean_words(text)

        if not words:
            return []

        if len(words) >= 2:

            results = self.next_words_from_phrase(
                f"{words[-2]} {words[-1]}",
                limit=limit,
                source=source,
            )

            if results:
                return results

        return self.next_words(
            words[-1],
            limit=limit,
            source=source,
        )

    # -------------------------------------------------------------------
    # SOURCE COMPARISON
    # -------------------------------------------------------------------

    def compare_sources(
        self,
        word: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare how different knowledge sources understand the continuation
        of a word.

        This is useful for GridCV and retrieval.

        Example sources:

            personal
            project
            workspace
            external
            dictionary
            ebook
            library
        """

        result = {}

        for source in self.source_chains.keys():

            result[source] = self.next_words(
                word,
                limit=limit,
                source=source,
            )

        return result

    # -------------------------------------------------------------------
    # KNOWLEDGE PROFILE
    # -------------------------------------------------------------------

    def knowledge_profile(
        self,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return statistics describing the current WordChain.
        """

        if source:
            source = self.normalize_source(source)

            chain = self.source_chains[source]

            return {
                "source": source,
                "unique_words": len(
                    chain["words"]
                ),
                "unique_pairs": len(
                    chain["pairs"]
                ),
                "unique_transitions": len(
                    chain["transitions"]
                ),
                "documents": self.source_statistics[
                    source
                ]["documents"],
                "words": self.source_statistics[
                    source
                ]["words"],
                "pairs": self.source_statistics[
                    source
                ]["pairs"],
            }

        return {
            "unique_words": len(
                self.word_frequency
            ),
            "unique_pairs": len(
                self.pairs
            ),
            "unique_transitions": len(
                self.transitions
            ),
            "sources": {
                source: dict(stats)
                for source, stats
                in self.source_statistics.items()
            },
        }

    # -------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize WordChain state.

        Useful for persistent personal AI state.
        """

        source_chains = {}

        for source, chain in self.source_chains.items():

            source_chains[source] = {
                "words": dict(
                    chain["words"]
                ),
                "pairs": dict(
                    chain["pairs"]
                ),
                "transitions": {
                    word: dict(counter)
                    for word, counter
                    in chain["transitions"].items()
                },
                "phrases": {
                    phrase: dict(counter)
                    for phrase, counter
                    in chain["phrases"].items()
                },
            }

        return {
            "word_frequency": dict(
                self.word_frequency
            ),
            "pairs": dict(
                self.pairs
            ),
            "transitions": {
                word: dict(counter)
                for word, counter
                in self.transitions.items()
            },
            "phrase_chains": {
                phrase: dict(counter)
                for phrase, counter
                in self.phrase_chains.items()
            },
            "source_statistics": {
                source: dict(stats)
                for source, stats
                in self.source_statistics.items()
            },
            "source_chains": source_chains,
        }

    # -------------------------------------------------------------------
    # DESERIALIZATION
    # -------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "WordChain":
        """
        Restore a WordChain from serialized state.
        """

        chain = cls()

        chain.word_frequency.update(
            data.get(
                "word_frequency",
                {},
            )
        )

        chain.pairs.update(
            data.get(
                "pairs",
                {},
            )
        )

        for word, values in data.get(
            "transitions",
            {},
        ).items():

            chain.transitions[word].update(
                values
            )

        for phrase, values in data.get(
            "phrase_chains",
            {},
        ).items():

            chain.phrase_chains[phrase].update(
                values
            )

        for source, values in data.get(
            "source_statistics",
            {},
        ).items():

            chain.source_statistics[
                source
            ].update(values)

        for source, values in data.get(
            "source_chains",
            {},
        ).items():

            source_chain = chain.source_chains[
                source
            ]

            source_chain["words"].update(
                values.get("words", {})
            )

            source_chain["pairs"].update(
                values.get("pairs", {})
            )

            for word, transitions in values.get(
                "transitions",
                {},
            ).items():

                source_chain[
                    "transitions"
                ][word].update(
                    transitions
                )

            for phrase, transitions in values.get(
                "phrases",
                {},
            ).items():

                source_chain[
                    "phrases"
                ][phrase].update(
                    transitions
                )

        return chain

    # -------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------

    def save(
        self,
        path: str,
    ) -> None:
        """
        Save the chain to JSON.
        """

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    # -------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "WordChain":
        """
        Load an existing WordChain.

        If the file does not exist, return an empty chain.
        """

        if not os.path.exists(path):
            return cls()

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        return cls.from_dict(data)

    # -------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------

    @staticmethod
    def _probability(
        count: int,
        total: int,
    ) -> float:
        """
        Calculate a local transition probability.
        """

        if total <= 0:
            return 0.0

        return count / total


# ---------------------------------------------------------------------------
# DEFAULT FACTORY
# ---------------------------------------------------------------------------

def create_word_chain(
    path: Optional[str] = None,
) -> WordChain:
    """
    Create or restore a WordChain.

    If path is supplied and exists, the chain is restored.
    """

    if path:
        return WordChain.load(path)

    return WordChain()


# ---------------------------------------------------------------------------
# SIMPLE MODULE-LEVEL HELPERS
# ---------------------------------------------------------------------------

def extract_word_pairs(
    text: str,
) -> List[str]:
    """
    Compatibility helper for existing modules such as follow_up.py
    and ranking.py.
    """

    words = clean_words(text)

    return [
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    ]


def extract_next_word_candidates(
    text: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compatibility helper.

    Builds a temporary chain from supplied text.
    """

    chain = WordChain()
    chain.add_text(
        text,
        source="conversation",
    )

    results = []

    for word in clean_words(text):

        candidates = chain.next_words(
            word,
            limit=limit,
        )

        results.extend(candidates)

    # Remove duplicate transitions while preserving strongest counts.
    seen = set()
    unique = []

    for result in sorted(
        results,
        key=lambda x: x["count"],
        reverse=True,
    ):

        key = (
            result["after"],
            result["word"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

        if len(unique) >= limit:
            break

    return unique


# ---------------------------------------------------------------------------
# TEST / DEVELOPMENT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    chain = WordChain()

    # Personal conversation knowledge.
    chain.add_text(
        "My project uses GSP for deterministic storage.",
        source="project",
    )

    chain.add_text(
        "GSP provides deterministic storage for the project.",
        source="project",
    )

    # External knowledge.
    chain.add_text(
        "A deterministic system produces the same result for the same input.",
        source="dictionary",
    )

    print(
        "Project continuation:"
    )

    print(
        chain.next_words(
            "gsp",
            limit=5,
            source="project",
        )
    )

    print(
        "\nPhrase continuation:"
    )

    print(
        chain.next_words_from_phrase(
            "deterministic storage",
            limit=5,
        )
    )

    print(
        "\nPair overlap:"
    )

    print(
        chain.pair_overlap(
            "GSP deterministic storage",
            "The project uses GSP for deterministic storage.",
        )
    )

    print(
        "\nKnowledge profile:"
    )

    print(
        chain.knowledge_profile()
    )