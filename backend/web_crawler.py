"""
CoMpaNeoN Web Crawler
=====================

External knowledge acquisition layer.

ARCHITECTURE
------------

                    external.py
                         │
              External source directions
                         │
                         ▼
                    WebCrawler
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        Web          YouTube        ApiTube/API
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                 Content extraction
                         │
                 checksum + metadata
                         │
                         ▼
                    tokenizer.py
                         │
                         ▼
                    MemoryGrid
                         │
                         ▼
                 GridCrawler / GridCV
                         │
                         ▼
                  WordChain + Ranking
                         │
                         ▼
                 WordUnderstanding
                         │
                         ▼
                   PromptManager

RESPONSIBILITIES
----------------

WebCrawler:

    - acquire external information
    - follow source directions
    - fetch HTML/API/video metadata
    - extract readable text
    - acquire subtitles/transcripts when available
    - calculate content checksums
    - detect unchanged content
    - attach source metadata
    - tokenize acquired text
    - send documents into MemoryGrid
    - expose crawl statistics

WebCrawler does NOT own:

    - GSP mathematics
    - alphabet mathematics
    - Word Grid mathematics
    - ranking
    - WordChain
    - WordUnderstanding
    - PromptManager
    - AI response generation
    - MemoryGrid placement mathematics

The canonical linguistic authority remains tokenizer.py.

The canonical storage/indexing authority remains MemoryGrid.

The crawler is an acquisition layer, not a competing database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# OPTIONAL EXTERNAL DIRECTIONS
# ---------------------------------------------------------------------------

try:
    from external import (
        fetch_dictionary,
        fetch_news,
        fetch_books,
        fetch_elibrary,
        fetch_wikipedia,
        fetch_github_ebooks,
        fetch_code_textbook,
        fetch_alphavantage,
        fetch_financial_modelling_prep,
        fetch_youtube,
        fetch_apitube,
    )
except ImportError:
    # The crawler can still operate as a normal HTTP crawler if
    # external.py has not yet exposed every source adapter.
    fetch_dictionary = None
    fetch_news = None
    fetch_books = None
    fetch_elibrary = None
    fetch_wikipedia = None
    fetch_github_ebooks = None
    fetch_code_textbook = None
    fetch_alphavantage = None
    fetch_financial_modelling_prep = None
    fetch_youtube = None
    fetch_apitube = None


class WebCrawler:
    """
    CoMpaNeoN external knowledge acquisition layer.

    A WebCrawler instance receives a shared MemoryGrid and feeds
    acquired external information into that grid.

    Source-specific acquisition can be delegated to external.py.

    Example:

        crawler = WebCrawler(memory_grid)

        crawler.crawl(
            "https://example.com"
        )

    Or:

        crawler.acquire_external(
            source="wikipedia",
            query="quantum mechanics",
        )
    """

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        memory_grid,
        timeout: float = 15.0,
        user_agent: str = "CoMpaNeoN-WebCrawler/1.0",
    ) -> None:

        self.memory = memory_grid

        self.page_cache = PageCache()

        self.scheduler = CrawlerScheduler()

        self.timeout = float(timeout)

        self.user_agent = user_agent

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        self.pages_crawled = 0
        self.pages_cached = 0

        self.documents_indexed = 0
        self.tokens_indexed = 0

        self.external_requests = 0
        self.external_documents = 0

        self.video_sources = 0
        self.transcripts_acquired = 0

        # --------------------------------------------------------------
        # Source registry
        # --------------------------------------------------------------

        self.external_sources = {
            "dictionary": fetch_dictionary,
            "news": fetch_news,
            "books": fetch_books,
            "elibrary": fetch_elibrary,
            "wikipedia": fetch_wikipedia,
            "github_ebooks": fetch_github_ebooks,
            "code_textbook": fetch_code_textbook,
            "alphavantage": fetch_alphavantage,
            "financial_modeling_prep": (
                fetch_financial_modelling_prep
            ),
            "youtube": fetch_youtube,
            "apitube": fetch_apitube,
        }

    # ======================================================================
    # TIME
    # ======================================================================

    @staticmethod
    def _timestamp() -> str:
        """
        Return a deterministic UTC acquisition timestamp.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ======================================================================
    # HASH
    # ======================================================================

    @staticmethod
    def content_hash(
        text: str,
    ) -> str:
        """
        SHA-256 checksum of normalized content.

        Used for:

            - duplicate detection
            - change detection
            - source verification
            - memory indexing metadata
        """

        normalized = (
            str(text)
            .strip()
            .replace(
                "\r\n",
                "\n",
            )
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

    # ======================================================================
    # URL HASH
    # ======================================================================

    @staticmethod
    def source_hash(
        url: str,
    ) -> str:
        """
        Deterministic source identifier.
        """

        return hashlib.sha256(
            str(url)
            .strip()
            .encode("utf-8")
        ).hexdigest()

    # ======================================================================
    # HTTP HEADERS
    # ======================================================================

    def _headers(
        self,
    ) -> Dict[str, str]:

        return {
            "User-Agent": self.user_agent,
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/json,"
                "text/plain"
            ),
        }

    # ======================================================================
    # FETCH WEB PAGE
    # ======================================================================

    def fetch_text(
        self,
        url: str,
    ) -> str:
        """
        Fetch a normal web page and extract visible text.
        """

        response = httpx.get(
            url,
            timeout=self.timeout,
            follow_redirects=True,
            headers=self._headers(),
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "template",
                "svg",
                "canvas",
            ]
        ):
            element.decompose()

        return soup.get_text(
            separator=" ",
            strip=True,
        )

    # ======================================================================
    # FETCH RAW JSON
    # ======================================================================

    def fetch_json(
        self,
        url: str,
        params: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Fetch a JSON endpoint.

        Used when a source exposes structured metadata rather than HTML.
        """

        response = httpx.get(
            url,
            params=params,
            timeout=self.timeout,
            follow_redirects=True,
            headers=self._headers(),
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            return data

        return {
            "data": data
        }

    # ======================================================================
    # LANGUAGE
    # ======================================================================

    def _resolve_language(
        self,
        lang: Optional[str],
    ) -> str:

        return normalize_lang(
            lang or "en"
        )

    # ======================================================================
    # LANGUAGE DETECTION
    # ======================================================================

    def detect_language(
        self,
        text: str,
        fallback: str = "en",
    ) -> str:
        """
        Detect language when possible.

        langdetect remains optional so that crawler acquisition does not
        become dependent on it during environments where the package has
        not yet been installed.
        """

        try:

            from langdetect import detect

            if not text.strip():
                return fallback

            detected = detect(
                text
            )

            return normalize_lang(
                detected
            )

        except Exception:

            return self._resolve_language(
                fallback
            )

    # ======================================================================
    # NORMALIZE TEXT
    # ======================================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace while preserving textual content.
        """

        text = str(text)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ======================================================================
    # TOKENIZATION
    # ======================================================================

    def index_words(
        self,
        text: str,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Send acquired content through the canonical tokenizer.

        WebCrawler does not calculate:

            letter indices
            alphabet dimensions
            Word Grid coordinates
            GSP cells

        tokenizer.py owns those responsibilities.
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

    # ======================================================================
    # METADATA
    # ======================================================================

    def build_metadata(
        self,
        *,
        url: str = "",
        source_type: str = "web",
        lang: str = "en",
        title: str = "",
        content_hash: str = "",
        content_type: str = "text",
        author: str = "",
        published_at: Optional[str] = None,
        duration: Optional[float] = None,
        subtitles: bool = False,
        transcript: bool = False,
        extra: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        metadata: Dict[str, Any] = {

            "source": source_type,

            "source_url": url,

            "source_id": (
                self.source_hash(url)
                if url
                else ""
            ),

            "title": title,

            "language": lang,

            "content_type": content_type,

            "content_hash": content_hash,

            "author": author,

            "published_at": published_at,

            "duration": duration,

            "subtitles_available": subtitles,

            "transcript_available": transcript,

            "acquired_at": self._timestamp(),

            "crawler": "CoMpaNeoN-WebCrawler",

            "crawler_version": "1.0",

        }

        if extra:
            metadata.update(
                extra
            )

        return metadata

    # ======================================================================
    # MEMORYGRID INDEXING
    # ======================================================================

    def index_into_memory(
        self,
        text: str,
        url: str = "",
        lang: str = "en",
        source_type: Any = "web",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Index acquired content into MemoryGrid.

        MemoryGrid remains the canonical storage/indexing authority.
        """

        text = self.normalize_text(
            text
        )

        if not text:
            return {
                "indexed": False,
                "reason": "empty_content",
            }

        lang = self._resolve_language(
            lang
        )

        token_data = self.index_words(
            text,
            lang,
        )

        source_label = (
            str(source_type)
        )

        if url:
            source_label = (
                f"{source_label}:{url}"
            )

        # --------------------------------------------------------------
        # MemoryGrid owns actual placement.
        # --------------------------------------------------------------

        doc_id = self.memory.add_document(
            text=text,
            lang=lang,
            source=source_label,
        )

        self.documents_indexed += 1

        self.tokens_indexed += (
            token_data["token_count"]
        )

        return {
            "indexed": True,
            "doc_id": doc_id,
            "language": lang,
            "token_count": token_data[
                "token_count"
            ],
            "tokens": token_data[
                "tokens"
            ],
            "metadata": metadata or {},
        }

    # ======================================================================
    # DOCUMENT ACQUISITION
    # ======================================================================

    def acquire_document(
        self,
        text: str,
        *,
        url: str = "",
        lang: Optional[str] = None,
        source_type: str = "web",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Common acquisition path for every external source.

        All sources eventually converge here.
        """

        text = self.normalize_text(
            text
        )

        if not text:
            return {
                "indexed": False,
                "text": "",
            }

        resolved_lang = (
            self.detect_language(
                text
            )
            if lang is None
            else self._resolve_language(
                lang
            )
        )

        checksum = self.content_hash(
            text
        )

        final_metadata = (
            metadata
            or self.build_metadata(
                url=url,
                source_type=source_type,
                lang=resolved_lang,
                content_hash=checksum,
            )
        )

        result = self.index_into_memory(
            text=text,
            url=url,
            lang=resolved_lang,
            source_type=source_type,
            metadata=final_metadata,
        )

        result.update({
            "url": url,
            "source_type": source_type,
            "content_hash": checksum,
        })

        return result

    # ======================================================================
    # CRAWL ONE WEB PAGE
    # ======================================================================

    def crawl(
        self,
        url: str,
        source_type: Any = ContentType.NORMAL_WEB,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crawl one external web page.

        Pipeline:

            URL
             ↓
            PageCache
             ↓
            HTTP
             ↓
            HTML extraction
             ↓
            checksum
             ↓
            language detection
             ↓
            tokenizer
             ↓
            MemoryGrid
        """

        requested