"""
CoMpaNeoN Web Crawler

External knowledge acquisition and MemoryGrid ingestion layer.

ARCHITECTURE

                external.py
                     │
          External source directions
                     │
                     ▼
             CrawlerScheduler
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
      ┌──────────────┼──────────────┐
      │              │              │
   Letter Grid    Word Grid     Storage Grid
                                       │
                                       ▼
                               GridCrawler / GridCV
                                       │
                                       ▼
                                CrawlerRetrieval
                                       │
                                       ▼
                               Higher AI retrieval

RESPONSIBILITIES

WebCrawler:

- acquire external information
- accept scheduler directions
- fetch HTML/API/video metadata
- extract readable text
- acquire subtitles/transcripts when available
- calculate content checksums
- detect unchanged content
- attach source metadata
- resolve language
- send complete documents into MemoryGrid
- coordinate scheduled crawling
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

IMPORTANT

MemoryGrid remains the canonical indexing authority.

WebCrawler acquires.

MemoryGrid tokenizes and places.

CrawlerScheduler determines when crawling occurs.

Future MemoryGrid orchestration may instantiate:

- WebCrawler
- GridCrawler
- GridCV
- CrawlerRetrieval

without changing the ownership boundaries above.
"""

from future import annotations

import hashlib
import inspect
import re

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

============================================================================

INTERNAL DEPENDENCIES

============================================================================

from page_cache import PageCache

from crawler_scheduler import (
CrawlerScheduler,
ContentType,
)

from tokenizer import (
normalize_lang,
tokenize,
)

============================================================================

OPTIONAL EXTERNAL SOURCE ADAPTERS

============================================================================

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

============================================================================

WEB CRAWLER

============================================================================

class WebCrawler:
"""
CoMpaNeoN external knowledge acquisition layer.

The WebCrawler receives a shared MemoryGrid.

It does not own a separate knowledge store.

All acquired knowledge converges into:

    MemoryGrid.add_document()

Example
-------

    crawler = WebCrawler(memory_grid)

    crawler.crawl(
        "https://example.com"
    )

Scheduled:

    crawler.schedule(
        "https://example.com"
    )

External:

    await crawler.acquire_external(
        source="wikipedia",
        query="quantum mechanics",
    )
"""

# ========================================================================
# INITIALIZATION
# ========================================================================

def __init__(
    self,
    memory_grid: Any,
    timeout: float = 15.0,
    user_agent: str = (
        "CoMpaNeoN-WebCrawler/1.0"
    ),
    scheduler: Optional[
        CrawlerScheduler
    ] = None,
) -> None:

    # --------------------------------------------------------------------
    # SHARED MEMORY
    # --------------------------------------------------------------------

    self.memory = memory_grid

    # --------------------------------------------------------------------
    # CACHE
    # --------------------------------------------------------------------

    self.page_cache = PageCache()

    # --------------------------------------------------------------------
    # SCHEDULER
    #
    # A scheduler may be injected by MemoryGrid or a higher-level
    # crawler orchestration system.
    #
    # Otherwise a local scheduler is created.
    # --------------------------------------------------------------------

    self.scheduler = (
        scheduler
        if scheduler is not None
        else CrawlerScheduler()
    )

    # --------------------------------------------------------------------
    # NETWORK
    # --------------------------------------------------------------------

    self.timeout = float(
        timeout
    )

    self.user_agent = (
        str(user_agent)
    )

    # --------------------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------------------

    self.pages_crawled = 0

    self.pages_cached = 0

    self.pages_unchanged = 0

    self.documents_indexed = 0

    self.tokens_indexed = 0

    self.external_requests = 0

    self.external_documents = 0

    self.video_sources = 0

    self.transcripts_acquired = 0

    self.scheduled_jobs = 0

    self.scheduler_runs = 0

    # --------------------------------------------------------------------
    # SOURCE REGISTRY
    # --------------------------------------------------------------------

    self.external_sources = {

        "dictionary":
            fetch_dictionary,

        "news":
            fetch_news,

        "books":
            fetch_books,

        "elibrary":
            fetch_elibrary,

        "wikipedia":
            fetch_wikipedia,

        "github_ebooks":
            fetch_github_ebooks,

        "code_textbook":
            fetch_code_textbook,

        "alphavantage":
            fetch_alphavantage,

        "financial_modeling_prep":
            fetch_financial_modelling_prep,

        "youtube":
            fetch_youtube,

        "apitube":
            fetch_apitube,

    }

# ========================================================================
# TIME
# ========================================================================

@staticmethod
def _timestamp() -> str:
    """
    Return a UTC acquisition timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()

# ========================================================================
# HASH
# ========================================================================

@staticmethod
def content_hash(
    text: str,
) -> str:
    """
    Return SHA-256 checksum of normalized content.
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

# ========================================================================
# SOURCE HASH
# ========================================================================

@staticmethod
def source_hash(
    url: str,
) -> str:
    """
    Return deterministic source identifier.
    """

    return hashlib.sha256(
        str(url)
        .strip()
        .encode(
            "utf-8"
        )
    ).hexdigest()

# ========================================================================
# HEADERS
# ========================================================================

def _headers(
    self,
) -> Dict[str, str]:

    return {

        "User-Agent":
            self.user_agent,

        "Accept": (
            "text/html,"
            "application/xhtml+xml,"
            "application/json,"
            "text/plain"
        ),

    }

# ========================================================================
# LANGUAGE
# ========================================================================

def _resolve_language(
    self,
    lang: Optional[str],
) -> str:
    """
    Normalize a language code through tokenizer.py.
    """

    return normalize_lang(
        lang or "en"
    )

# ========================================================================
# LANGUAGE DETECTION
# ========================================================================

def detect_language(
    self,
    text: str,
    fallback: str = "en",
) -> str:
    """
    Detect language when optional language detection support
    is available.
    """

    if not str(
        text
    ).strip():

        return self._resolve_language(
            fallback
        )

    try:

        from langdetect import detect

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

# ========================================================================
# TEXT NORMALIZATION
# ========================================================================

@staticmethod
def normalize_text(
    text: str,
) -> str:
    """
    Normalize whitespace.

    Linguistic normalization remains owned by tokenizer.py.
    """

    text = str(
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

# ========================================================================
# FETCH HTML
# ========================================================================

def fetch_text(
    self,
    url: str,
) -> str:
    """
    Fetch a web page and extract visible readable text.
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

# ========================================================================
# FETCH JSON
# ========================================================================

def fetch_json(
    self,
    url: str,
    params: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Fetch structured JSON.
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

    if isinstance(
        data,
        dict,
    ):

        return data

    return {
        "data": data
    }

# ========================================================================
# TOKENIZATION DIAGNOSTICS
# ========================================================================

def inspect_tokens(
    self,
    text: str,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Tokenize for crawler statistics and diagnostics.

    IMPORTANT:

    MemoryGrid remains responsible for canonical indexing.

    This method does not independently place tokens.
    """

    lang = self._resolve_language(
        lang
    )

    tokens = tokenize(
        text,
        lang,
    )

    return {

        "language":
            lang,

        "tokens":
            tokens,

        "token_count":
            len(tokens),

    }

# ========================================================================
# METADATA
# ========================================================================

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
    published_at: Optional[
        str
    ] = None,
    duration: Optional[
        float
    ] = None,
    subtitles: bool = False,
    transcript: bool = False,
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build canonical acquisition metadata.
    """

    metadata: Dict[
        str,
        Any
    ] = {

        "source":
            source_type,

        "source_url":
            url,

        "source_id":
            (
                self.source_hash(
                    url
                )
                if url
                else ""
            ),

        "title":
            title,

        "language":
            lang,

        "content_type":
            content_type,

        "content_hash":
            content_hash,

        "author":
            author,

        "published_at":
            published_at,

        "duration":
            duration,

        "subtitles_available":
            subtitles,

        "transcript_available":
            transcript,

        "acquired_at":
            self._timestamp(),

        "crawler":
            "CoMpaNeoN-WebCrawler",

        "crawler_version":
            "1.0",

    }

    if extra:

        metadata.update(
            extra
        )

    return metadata

# ========================================================================
# MEMORY INGESTION
# ========================================================================

def index_into_memory(
    self,
    text: str,
    *,
    url: str = "",
    lang: str = "en",
    source_type: Any = "web",
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Send acquired content into MemoryGrid.

    MemoryGrid remains the canonical owner of:

        - tokenization
        - Letter Grid placement
        - Word Grid placement
        - full-text storage placement
        - document identity

    WebCrawler supplies acquisition context.
    """

    text = self.normalize_text(
        text
    )

    if not text:

        return {

            "indexed":
                False,

            "reason":
                "empty_content",

        }

    lang = self._resolve_language(
        lang
    )

    # --------------------------------------------------------------------
    # DIAGNOSTIC TOKEN COUNT
    #
    # MemoryGrid will tokenize again during canonical indexing.
    #
    # This can later be optimized when MemoryGrid accepts a verified
    # tokenizer payload directly.
    # --------------------------------------------------------------------

    token_data = self.inspect_tokens(
        text,
        lang,
    )

    source_label = str(
        source_type
    )

    if url:

        source_label = (
            f"{source_label}:{url}"
        )

    # --------------------------------------------------------------------
    # CANONICAL MEMORYGRID INSERTION
    # --------------------------------------------------------------------

    try:

        doc_id = self.memory.add_document(
            text=text,
            lang=lang,
            source=source_label,
            metadata=metadata or {},
        )

    except TypeError:

        # Compatibility with the current MemoryGrid signature.
        #
        # The upcoming MemoryGrid rewrite should accept metadata.
        # Until then, the crawler remains compatible.

        doc_id = self.memory.add_document(
            text=text,
            lang=lang,
            source=source_label,
        )

    self.documents_indexed += 1

    self.tokens_indexed += (
        token_data[
            "token_count"
        ]
    )

    return {

        "indexed":
            True,

        "doc_id":
            doc_id,

        "language":
            lang,

        "token_count":
            token_data[
                "token_count"
            ],

        "metadata":
            metadata or {},

    }

# ========================================================================
# COMMON DOCUMENT ACQUISITION
# ========================================================================

def acquire_document(
    self,
    text: str,
    *,
    url: str = "",
    lang: Optional[
        str
    ] = None,
    source_type: str = "web",
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Common ingestion route for every external source.

    Every source eventually converges here.
    """

    text = self.normalize_text(
        text
    )

    if not text:

        return {

            "indexed":
                False,

            "text":
                "",

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

    final_metadata.setdefault(
        "content_hash",
        checksum,
    )

    final_metadata.setdefault(
        "source_url",
        url,
    )

    result = self.index_into_memory(
        text=text,
        url=url,
        lang=resolved_lang,
        source_type=source_type,
        metadata=final_metadata,
    )

    result.update({

        "url":
            url,

        "source_type":
            source_type,

        "content_hash":
            checksum,

    })

    return result

# ========================================================================
# CRAWL ONE PAGE
# ========================================================================

def crawl(
    self,
    url: str,
    source_type: Any = (
        ContentType.NORMAL_WEB
    ),
    lang: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    """
    Crawl one web resource and ingest changed content into MemoryGrid.

    Pipeline:

        URL
         ↓
        PageCache
         ↓
        HTTP
         ↓
        Content extraction
         ↓
        Checksum
         ↓
        Change detection
         ↓
        Language resolution
         ↓
        Metadata
         ↓
        MemoryGrid
    """

    requested_lang = (

        self._resolve_language(
            lang
        )

        if lang

        else None

    )

    # --------------------------------------------------------------------
    # FETCH
    # --------------------------------------------------------------------

    try:

        text = self.fetch_text(
            url
        )

    except Exception as exc:

        return {

            "url":
                url,

            "cached":
                False,

            "changed":
                False,

            "indexed":
                False,

            "error":
                str(exc),

        }

    text = self.normalize_text(
        text
    )

    if not text:

        return {

            "url":
                url,

       