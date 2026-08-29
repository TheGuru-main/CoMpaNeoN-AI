"""
CoMpaNeoN Web Crawler
=====================

External knowledge acquisition layer.

Architecture
------------

WebCrawler
    ↓
Web / external source
    ↓
Tokenizer
    ↓
MemoryGrid
    ↓
GridCrawler / GridScheduler
    ↓
AI training / retrieval

The WebCrawler does NOT own:

    - GSP placement mathematics
    - Word Grid placement
    - MemoryGrid retrieval
    - GridCrawler traversal
    - AI training
    - ranking
    - response generation

Its responsibility is to acquire external information, tokenize it,
and index it into the cloud-based MemoryGrid.

ARCHITECTURAL STORAGE MODEL
----------------------------

The cloud MemoryGrid is the persistent retrieval box used by the AI.

It can contain:

    1. Web-crawled knowledge
    2. User-entered knowledge
    3. AI-generated knowledge/output
    4. Training material
    5. Domain/dictionary knowledge
    6. Other indexed knowledge

The physical/user storage layer is separate.

After an AI instance is downloaded for a user or organization, its
physical storage endpoint may contain partitioned local representations
of:

    - MemoryGrid data
    - web-crawled data
    - STM/LTM data
    - AI-generated data
    - user/profile data

Those physical partitions remain deterministic and belong to that
user/organization's downloaded AI file.

The cloud MemoryGrid remains the continuously accessible knowledge
source.

IMPORTANT
---------

The WebCrawler does not independently create:

    col = ord(first_letter) - 97

That would duplicate and eventually diverge from tokenizer.py.

Instead:

    WebCrawler
        → tokenizer.py
        → MemoryGrid.add_document()

The tokenizer remains the canonical authority for:

    - language normalization
    - alphabet mapping
    - letter index
    - Word Grid coordinate
    - language-specific alphabet dimensions

The MemoryGrid remains the canonical cloud retrieval box.

The GridCrawler remains responsible for crawling through the grid.

The CrawlerScheduler remains responsible for scheduling crawler work.

The WebCrawler therefore feeds the MemoryGrid rather than maintaining
a separate competing word index.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from bs4 import BeautifulSoup

from page_cache import PageCache
from crawler_scheduler import (
    CrawlerScheduler,
    ContentType,
)

from tokenizer import (
    normalize_lang,
    tokenize,
)


class WebCrawler:
    """
    External web knowledge acquisition layer.

    The crawler acquires external text and places the resulting
    knowledge into the cloud-based MemoryGrid.

    Parameters
    ----------
    memory_grid:
        Shared MemoryGrid instance.

    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        memory_grid,
        timeout: float = 10.0,
    ) -> None:

        self.memory = memory_grid

        self.page_cache = PageCache()

        self.scheduler = CrawlerScheduler()

        self.timeout = float(
            timeout
        )

        # --------------------------------------------------------------
        # Crawl statistics
        # --------------------------------------------------------------

        self.pages_crawled = 0

        self.pages_cached = 0

        self.documents_indexed = 0

        self.tokens_indexed = 0

    # ==================================================================
    # FETCH
    # ==================================================================

    def fetch_text(
        self,
        url: str,
    ) -> str:
        """
        Fetch a web page and extract visible textual content.

        HTML structure is removed before the content enters the
        tokenizer / MemoryGrid pipeline.
        """

        response = httpx.get(
            url,
            timeout=self.timeout,
            follow_redirects=True,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Remove elements that should not become knowledge tokens.

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "template",
            ]
        ):
            element.decompose()

        return soup.get_text(
            separator=" ",
            strip=True,
        )

    # ==================================================================
    # LANGUAGE
    # ==================================================================

    def _resolve_language(
        self,
        lang: Optional[str],
    ) -> str:
        """
        Normalize the language before tokenization.
        """

        return normalize_lang(
            lang or "en"
        )

    # ==================================================================
    # TOKENIZATION / INDEXING
    # ==================================================================

    def index_words(
        self,
        text: str,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Tokenize external text using the canonical tokenizer.

        No independent alphabet calculation is performed here.

        The tokenizer determines:

            letter indices
            Word Grid positions
            language alphabet
            word length
            stems

        The resulting document is then handed to MemoryGrid.
        """

        lang = self._resolve_language(
            lang
        )

        tokens = tokenize(
            text,
            lang,
        )

        return {
            "language": lang,
            "tokens": tokens,
            "token_count": len(tokens),
        }

    # ==================================================================
    # MEMORYGRID INDEXING
    # ==================================================================

    def index_into_memory(
        self,
        text: str,
        url: str,
        lang: str = "en",
        source_type: ContentType = ContentType.NORMAL_WEB,
    ) -> Dict[str, Any]:
        """
        Store crawled knowledge in the shared cloud MemoryGrid.

        MemoryGrid owns the actual indexing routes.

        WebCrawler does not recreate:

            Letter Grid
            Word Grid
            Storage Grid
            GSP placement
        """

        lang = self._resolve_language(
            lang
        )

        token_data = self.index_words(
            text,
            lang,
        )

        doc_id = self.memory.add_document(
            text=text,
            lang=lang,
            source=f"{source_type}:{url}",
        )

        token_count = token_data[
            "token_count"
        ]

        self.documents_indexed += 1

        self.tokens_indexed += token_count

        return {
            "doc_id": doc_id,
            "url": url,
            "source_type": str(
                source_type
            ),
            "language": lang,
            "token_count": token_count,
            "tokens": token_data[
                "tokens"
            ],
        }

    # ==================================================================
    # CRAWL
    # ==================================================================

    def crawl(
        self,
        url: str,
        source_type: ContentType = ContentType.NORMAL_WEB,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Crawl one external page.

        Pipeline:

            URL
             ↓
            PageCache
             ↓
            HTTP fetch
             ↓
            HTML extraction
             ↓
            tokenizer.py
             ↓
            MemoryGrid
             ↓
            GridCrawler retrieval
        """

        lang = self._resolve_language(
            lang
        )

        # --------------------------------------------------------------
        # CACHE
        # --------------------------------------------------------------

        cached_data, cached_hash = (
            self.page_cache.get(url)
        )

        if cached_data is not None:

            self.pages_cached += 1

            return {
                "url": url,
                "cached": True,
                "changed": False,
                "language": lang,
                "text": cached_data,
                "content_hash": cached_hash,
                "indexed": False,
            }

        # --------------------------------------------------------------
        # FETCH
        # --------------------------------------------------------------

        text = self.fetch_text(
            url
        )

        if not text:
            return {
                "url": url,
                "cached": False,
                "changed": False,
                "language": lang,
                "text": "",
                "indexed": False,
            }

        self.pages_crawled += 1

        # --------------------------------------------------------------
        # CONTENT HASH
        # --------------------------------------------------------------

        content_hash = (
            self.page_cache._hash(
                text
            )
        )

        # --------------------------------------------------------------
        # CHANGE DETECTION
        # --------------------------------------------------------------

        changed = (
            self.page_cache.has_changed(
                url,
                text,
            )
        )

        if not changed:

            self.page_cache.set(
                url,
                content_hash,
                text,
            )

            return {
                "url": url,
                "cached": False,
                "changed": False,
                "language": lang,
                "text": text,
                "content_hash": content_hash,
                "indexed": False,
            }

        # --------------------------------------------------------------
        # INDEX INTO MEMORYGRID
        # --------------------------------------------------------------

        indexed = self.index_into_memory(
            text=text,
            url=url,
            lang=lang,
            source_type=source_type,
        )

        # --------------------------------------------------------------
        # CACHE
        # --------------------------------------------------------------

        self.page_cache.set(
            url,
            content_hash,
            text,
        )

        return {
            "url": url,
            "cached": False,
            "changed": True,
            "language": lang,
            "text": text,
            "content_hash": content_hash,
            "indexed": True,
            "doc_id": indexed[
                "doc_id"
            ],
            "token_count": indexed[
                "token_count"
            ],
        }

    # ==================================================================
    # CRAWL + SCHEDULE
    # ==================================================================

    def schedule(
        self,
        url: str,
        source_type: ContentType = ContentType.NORMAL_WEB,
        lang: str = "en",
    ) -> Any:
        """
        Submit an external source to the crawler scheduler.

        The scheduler controls when crawling occurs.
        """

        return self.scheduler.schedule(
            url=url,
            source_type=source_type,
            lang=self._resolve_language(
                lang
            ),
        )

    # ==================================================================
    # CRAWL MULTIPLE
    # ==================================================================

    def crawl_many(
        self,
        urls: list[str],
        source_type: ContentType = ContentType.NORMAL_WEB,
        lang: str = "en",
    ) -> list[Dict[str, Any]]:
        """
        Crawl multiple external sources sequentially.

        Parallel scheduling remains the responsibility of the
        scheduler rather than this acquisition layer.
        """

        results = []

        for url in urls:

            results.append(
                self.crawl(
                    url=url,
                    source_type=source_type,
                    lang=lang,
                )
            )

        return results

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def stats(
        self,
    ) -> Dict[str, Any]:
        """
        Return crawler acquisition statistics.
        """

        return {
            "pages_crawled": (
                self.pages_crawled
            ),
            "pages_cached": (
                self.pages_cached
            ),
            "documents_indexed": (
                self.documents_indexed
            ),
            "tokens_indexed": (
                self.tokens_indexed
            ),
        }