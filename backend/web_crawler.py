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

        requested_lang = (
            self._resolve_language(lang)
            if lang
            else None
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
                "language": (
                    requested_lang
                    or self.detect_language(
                        cached_data
                    )
                ),
                "text": cached_data,
                "content_hash": cached_hash,
                "indexed": False,
            }

        # --------------------------------------------------------------
        # FETCH
        # --------------------------------------------------------------

        try:

            text = self.fetch_text(
                url
            )

        except Exception as exc:

            return {
                "url": url,
                "cached": False,
                "changed": False,
                "indexed": False,
                "error": str(exc),
            }

        if not text:

            return {
                "url": url,
                "cached": False,
                "changed": False,
                "indexed": False,
                "text": "",
            }

        self.pages_crawled += 1

        text = self.normalize_text(
            text
        )

        # --------------------------------------------------------------
        # CHECKSUM
        # --------------------------------------------------------------

        checksum = self.content_hash(
            text
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
                checksum,
                text,
            )

            return {
                "url": url,
                "cached": False,
                "changed": False,
                "language": (
                    requested_lang
                    or self.detect_language(
                        text
                    )
                ),
                "text": text,
                "content_hash": checksum,
                "indexed": False,
            }

        resolved_lang = (
            requested_lang
            or self.detect_language(
                text
            )
        )

        # --------------------------------------------------------------
        # METADATA
        # --------------------------------------------------------------

        metadata = self.build_metadata(
            url=url,
            source_type=str(
                source_type
            ),
            lang=resolved_lang,
            content_hash=checksum,
            content_type="text/html",
        )

        # --------------------------------------------------------------
        # MEMORYGRID
        # --------------------------------------------------------------

        indexed = self.acquire_document(
            text,
            url=url,
            lang=resolved_lang,
            source_type=str(
                source_type
            ),
            metadata=metadata,
        )

        # --------------------------------------------------------------
        # CACHE
        # --------------------------------------------------------------

        self.page_cache.set(
            url,
            checksum,
            text,
        )

        return {
            "url": url,
            "cached": False,
            "changed": True,
            "language": resolved_lang,
            "text": text,
            "content_hash": checksum,
            "indexed": indexed.get(
                "indexed",
                False,
            ),
            "doc_id": indexed.get(
                "doc_id"
            ),
            "token_count": indexed.get(
                "token_count",
                0,
            ),
        }

    # ======================================================================
    # EXTERNAL SOURCE DISPATCH
    # ======================================================================

    async def acquire_external(
        self,
        source: str,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Acquire knowledge from an adapter defined by external.py.

        Example:

            await crawler.acquire_external(
                "wikipedia",
                "quantum mechanics",
            )

        The external adapter determines HOW to communicate with the
        external API.

        WebCrawler determines HOW the returned information enters
        MemoryGrid.
        """

        source_key = (
            str(source)
            .strip()
            .lower()
        )

        adapter = self.external_sources.get(
            source_key
        )

        if adapter is None:

            return {
                "source": source_key,
                "query": query,
                "indexed": False,
                "error": (
                    "No external adapter "
                    f"registered for {source_key}"
                ),
            }

        if not callable(adapter):

            return {
                "source": source_key,
                "query": query,
                "indexed": False,
                "error": (
                    f"Adapter unavailable: "
                    f"{source_key}"
                ),
            }

        self.external_requests += 1

        try:

            result = await adapter(
                query,
                **kwargs,
            )

        except TypeError:

            # Some adapters accept only the primary query.
            result = await adapter(
                query
            )

        except Exception as exc:

            return {
                "source": source_key,
                "query": query,
                "indexed": False,
                "error": str(exc),
            }

        documents = (
            self._extract_external_documents(
                result
            )
        )

        indexed_documents = []

        for document in documents:

            text = document.get(
                "text",
                ""
            )

            if not text:
                continue

            metadata = document.get(
                "metadata",
                {}
            )

            indexed = self.acquire_document(
                text=text,
                url=document.get(
                    "url",
                    ""
                ),
                lang=document.get(
                    "language"
                ),
                source_type=source_key,
                metadata=metadata,
            )

            indexed_documents.append(
                indexed
            )

        self.external_documents += (
            len(indexed_documents)
        )

        return {
            "source": source_key,
            "query": query,
            "documents": indexed_documents,
            "document_count": len(
                indexed_documents
            ),
            "indexed": bool(
                indexed_documents
            ),
        }

    # ======================================================================
    # EXTERNAL RESULT NORMALIZATION
    # ======================================================================

    def _extract_external_documents(
        self,
        result: Any,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Convert different API response shapes into one crawler
        document representation.

        This prevents each external API from creating its own storage
        pipeline.
        """

        documents: List[
            Dict[str, Any]
        ] = []

        if result is None:
            return documents

        # --------------------------------------------------------------
        # Dictionary response
        # --------------------------------------------------------------

        if isinstance(result, dict):

            # Generic text response.
            if result.get("text"):

                documents.append({
                    "text": result["text"],
                    "url": result.get(
                        "url",
                        ""
                    ),
                    "language": result.get(
                        "language"
                    ),
                    "metadata": result.get(
                        "metadata",
                        {}
                    ),
                })

            # Wikipedia / news style.
            for key in (
                "articles",
                "ebooks",
                "books",
                "elibrary",
                "code_books",
            ):

                items = result.get(
                    key,
                    []
                )

                if not isinstance(
                    items,
                    list,
                ):
                    continue

                for item in items:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    text_parts = []

                    for field in (
                        "title",
                        "description",
                        "extract",
                        "summary",
                        "content",
                        "text",
                        "author",
                    ):

                        value = item.get(
                            field
                        )

                        if value:
                            text_parts.append(
                                str(value)
                            )

                    if not text_parts:
                        continue

                    documents.append({
                        "text": "\n".join(
                            text_parts
                        ),
                        "url": (
                            item.get("url")
                            or item.get(
                                "infoLink",
                                ""
                            )
                        ),
                        "language": (
                            item.get(
                                "language"
                            )
                        ),
                        "metadata": {
                            "external_record": item
                        },
                    })

            # ----------------------------------------------------------
            # Video result
            # ----------------------------------------------------------

            videos = result.get(
                "videos",
                []
            )

            if isinstance(
                videos,
                list,
            ):

                for video in videos:

                    if not isinstance(
                        video,
                        dict,
                    ):
                        continue

                    documents.extend(
                        self._video_to_documents(
                            video
                        )
                    )

            # ----------------------------------------------------------
            # Transcript result
            # ----------------------------------------------------------

            transcript = result.get(
                "transcript"
            )

            if transcript:

                documents.append({
                    "text": str(
                        transcript
                    ),
                    "url": result.get(
                        "url",
                        ""
                    ),
                    "language": result.get(
                        "language"
                    ),
                    "metadata": {
                        "content_type": (
                            "video_transcript"
                        ),
                        "transcript": True,
                    },
                })

        # --------------------------------------------------------------
        # List response
        # --------------------------------------------------------------

        elif isinstance(
            result,
            list,
        ):

            for item in result:

                documents.extend(
                    self._extract_external_documents(
                        item
                    )
                )

        return documents

    # ======================================================================
    # VIDEO NORMALIZATION
    # ======================================================================

    def _video_to_documents(
        self,
        video: Dict[str, Any],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Convert video metadata and transcript/subtitle material into
        MemoryGrid documents.
        """

        self.video_sources += 1

        documents = []

        title = video.get(
            "title",
            ""
        )

        description = video.get(
            "description",
            ""
        )

        url = (
            video.get("url")
            or video.get(
                "video_url",
                ""
            )
        )

        language = video.get(
            "language"
        )

        metadata = {
            "content_type": "video",
            "video_id": video.get(
                "id"
            ),
            "channel": video.get(
                "channel"
            ),
            "author": video.get(
                "author"
            ),
            "published_at": video.get(
                "published_at"
            ),
            "duration": video.get(
                "duration"
            ),
            "thumbnail": video.get(
                "thumbnail"
            ),
        }

        # --------------------------------------------------------------
        # Video metadata itself
        # --------------------------------------------------------------

        metadata_text = "\n".join(
            part
            for part in (
                title,
                description,
            )
            if part
        )

        if metadata_text:

            documents.append({
                "text": metadata_text,
                "url": url,
                "language": language,
                "metadata": metadata,
            })

        # --------------------------------------------------------------
        # Subtitles
        # --------------------------------------------------------------

        subtitles = video.get(
            "subtitles"
        )

        if subtitles:

            subtitle_text = (
                self._normalize_subtitles(
                    subtitles
                )
            )

            if subtitle_text:

                self.transcripts_acquired += 1

                documents.append({
                    "text": subtitle_text,
                    "url": url,
                    "language": language,
                    "metadata": {
                        **metadata,
                        "content_type": (
                            "video_subtitles"
                        ),
                        "subtitles": True,
                    },
                })

        # --------------------------------------------------------------
        # Transcript
        # --------------------------------------------------------------

        transcript = video.get(
            "transcript"
        )

        if transcript:

            self.transcripts_acquired += 1

            documents.append({
                "text": str(
                    transcript
                ),
                "url": url,
                "language": language,
                "metadata": {
                    **metadata,
                    "content_type": (
                        "video_transcript"
                    ),
                    "transcript": True,
                },
            })

        return documents

    # ======================================================================
    # SUBTITLE NORMALIZATION
    # ======================================================================

    def _normalize_subtitles(
        self,
        subtitles: Any,
    ) -> str:
        """
        Normalize common subtitle structures.

        Supports:

            string
            list of strings
            list of dictionaries
        """

        if isinstance(
            subtitles,
            str,
        ):
            return self.normalize_text(
                subtitles
            )

        if not isinstance(
            subtitles,
            list,
        ):
            return ""

        lines = []

        for item in subtitles:

            if isinstance(
                item,
                str,
            ):

                lines.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                text = (
                    item.get("text")
                    or item.get(
                        "caption"
                    )
                    or item.get(
                        "content"
                    )
                )

                if text:
                    lines.append(
                        str(text)
                    )

        return self.normalize_text(
            " ".join(lines)
        )

    # ======================================================================
    # YOUTUBE
    # ======================================================================

    async def crawl_youtube(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Acquire YouTube material through external.py.

        The YouTube adapter handles the external API/source mechanics.
        WebCrawler handles normalization and MemoryGrid ingestion.
        """

        return await self.acquire_external(
            source="youtube",
            query=query,
            **kwargs,
        )

    # ======================================================================
    # APITUBE
    # ======================================================================

    async def crawl_apitube(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Acquire video material through ApiTube.
        """

        return await self.acquire_external(
            source="apitube",
            query=query,
            **kwargs,
        )
