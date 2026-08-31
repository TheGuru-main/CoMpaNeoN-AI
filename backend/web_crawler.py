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

            "cached":
                False,

            "changed":
                False,

            "indexed":
                False,

            "text":
                "",

        }

    self.pages_crawled += 1

    # --------------------------------------------------------------------
    # CHECKSUM
    # --------------------------------------------------------------------

    checksum = self.content_hash(
        text
    )

    # --------------------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------------------

    changed = (
        self.page_cache.has_changed(
            url,
            text,
        )
    )

    resolved_lang = (
        requested_lang
        or self.detect_language(
            text
        )
    )

    # --------------------------------------------------------------------
    # UNCHANGED CONTENT
    # --------------------------------------------------------------------

    if not changed:

        self.pages_unchanged += 1

        return {

            "url":
                url,

            "cached":
                True,

            "changed":
                False,

            "indexed":
                False,

            "language":
                resolved_lang,

            "content_hash":
                checksum,

        }

    # --------------------------------------------------------------------
    # METADATA
    # --------------------------------------------------------------------

    metadata = self.build_metadata(
        url=url,
        source_type=str(
            source_type
        ),
        lang=resolved_lang,
        content_hash=checksum,
        content_type="text/html",
    )

    # --------------------------------------------------------------------
    # MEMORYGRID INGESTION
    # --------------------------------------------------------------------

    indexed = self.acquire_document(
        text=text,
        url=url,
        lang=resolved_lang,
        source_type=str(
            source_type
        ),
        metadata=metadata,
    )

    # --------------------------------------------------------------------
    # CACHE UPDATE
    # --------------------------------------------------------------------

    self.page_cache.set(
        url,
        checksum,
        text,
    )

    return {

        "url":
            url,

        "cached":
            False,

        "changed":
            True,

        "language":
            resolved_lang,

        "content_hash":
            checksum,

        "indexed":
            indexed.get(
                "indexed",
                False,
            ),

        "doc_id":
            indexed.get(
                "doc_id"
            ),

        "token_count":
            indexed.get(
                "token_count",
                0,
            ),

    }

# ========================================================================
# SCHEDULING
# ========================================================================

def schedule(
    self,
    url: str,
    source_type: Any = (
        ContentType.NORMAL_WEB
    ),
    lang: str = "en",
) -> Any:
    """
    Register a crawl job with CrawlerScheduler.

    The scheduler owns timing.

    WebCrawler owns acquisition.

    MemoryGrid owns indexing.
    """

    self.scheduled_jobs += 1

    return self.scheduler.schedule(
        url=url,
        source_type=source_type,
        lang=self._resolve_language(
            lang
        ),
    )

# ========================================================================
# SCHEDULE MANY
# ========================================================================

def schedule_many(
    self,
    urls: List[str],
    source_type: Any = (
        ContentType.NORMAL_WEB
    ),
    lang: str = "en",
) -> List[Any]:
    """
    Register multiple URLs with CrawlerScheduler.
    """

    jobs = []

    for url in urls:

        jobs.append(
            self.schedule(
                url=url,
                source_type=source_type,
                lang=lang,
            )
        )

    return jobs

# ========================================================================
# SCHEDULER EXECUTION
# ========================================================================

def run_scheduled(
    self,
    limit: Optional[
        int
    ] = None,
) -> List[
    Dict[str, Any]
]:
    """
    Execute scheduled crawler work.

    CrawlerScheduler is intentionally flexible because its exact
    execution API may evolve.

    Supported scheduler patterns may include:

        next()
        next_job()
        get_next()
        pop()
        run()

    Each resolved URL is ultimately sent back through:

        WebCrawler.crawl()
            ↓
        MemoryGrid.add_document()
    """

    self.scheduler_runs += 1

    results = []

    processed = 0

    # --------------------------------------------------------------------
    # DIRECT SCHEDULER RUN API
    # --------------------------------------------------------------------

    if hasattr(
        self.scheduler,
        "run",
    ):

        try:

            scheduled_result = (
                self.scheduler.run()
            )

            if scheduled_result is not None:

                if isinstance(
                    scheduled_result,
                    list,
                ):

                    for item in scheduled_result:

                        results.extend(
                            self._execute_scheduled_item(
                                item
                            )
                        )

                        processed += 1

                        if (
                            limit is not None
                            and processed >= limit
                        ):

                            return results

                    return results

        except TypeError:

            pass

    # --------------------------------------------------------------------
    # QUEUE-STYLE SCHEDULER API
    # --------------------------------------------------------------------

    resolver = None

    for method_name in (

        "next_job",
        "get_next",
        "next",
        "pop",

    ):

        method = getattr(
            self.scheduler,
            method_name,
            None,
        )

        if callable(
            method
        ):

            resolver = method

            break

    if resolver is None:

        return results

    while True:

        if (
            limit is not None
            and processed >= limit
        ):

            break

        try:

            job = resolver()

        except Exception:

            break

        if job is None:

            break

        results.extend(
            self._execute_scheduled_item(
                job
            )
        )

        processed += 1

    return results

# ========================================================================
# EXECUTE SCHEDULED ITEM
# ========================================================================

def _execute_scheduled_item(
    self,
    job: Any,
) -> List[
    Dict[str, Any]
]:
    """
    Convert a scheduler item into WebCrawler work.
    """

    if job is None:

        return []

    if isinstance(
        job,
        str,
    ):

        return [

            self.crawl(
                url=job
            )

        ]

    if not isinstance(
        job,
        dict,
    ):

        return []

    url = (
        job.get("url")
        or job.get(
            "source_url"
        )
    )

    if not url:

        return []

    return [

        self.crawl(
            url=url,
            source_type=job.get(
                "source_type",
                ContentType.NORMAL_WEB,
            ),
            lang=job.get(
                "lang"
            ),
        )

    ]

# ========================================================================
# CRAWL MANY
# ========================================================================

def crawl_many(
    self,
    urls: List[str],
    source_type: Any = (
        ContentType.NORMAL_WEB
    ),
    lang: Optional[
        str
    ] = None,
) -> List[
    Dict[str, Any]
]:
    """
    Crawl multiple URLs directly.

    Use schedule_many() when the scheduler should determine
    execution order.
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

# ========================================================================
# EXTERNAL SOURCE DISPATCH
# ========================================================================

async def acquire_external(
    self,
    source: str,
    query: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Acquire knowledge through an external.py adapter.

    External adapters own communication with external sources.

    WebCrawler owns normalization and MemoryGrid ingestion.
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

            "source":
                source_key,

            "query":
                query,

            "indexed":
                False,

            "error":
                (
                    "No external adapter "
                    f"registered for "
                    f"{source_key}"
                ),

        }

    if not callable(
        adapter
    ):

        return {

            "source":
                source_key,

            "query":
                query,

            "indexed":
                False,

            "error":
                (
                    "Adapter unavailable: "
                    f"{source_key}"
                ),

        }

    self.external_requests += 1

    try:

        result = adapter(
            query,
            **kwargs,
        )

        if inspect.isawaitable(
            result
        ):

            result = await result

    except TypeError:

        try:

            result = adapter(
                query
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

        except Exception as exc:

            return {

                "source":
                    source_key,

                "query":
                    query,

                "indexed":
                    False,

                "error":
                    str(exc),

            }

    except Exception as exc:

        return {

            "source":
                source_key,

            "query":
                query,

            "indexed":
                False,

            "error":
                str(exc),

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
            metadata=document.get(
                "metadata",
                {}
            ),
        )

        indexed_documents.append(
            indexed
        )

    self.external_documents += len(
        indexed_documents
    )

    return {

        "source":
            source_key,

        "query":
            query,

        "documents":
            indexed_documents,

        "document_count":
            len(
                indexed_documents
            ),

        "indexed":
            bool(
                indexed_documents
            ),

    }

# ========================================================================
# EXTERNAL RESULT NORMALIZATION
# ========================================================================

def _extract_external_documents(
    self,
    result: Any,
) -> List[
    Dict[str, Any]
]:
    """
    Convert external API response shapes into crawler documents.
    """

    documents: List[
        Dict[str, Any]
    ] = []

    if result is None:

        return documents

    # --------------------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        if result.get(
            "text"
        ):

            documents.append({

                "text":
                    result[
                        "text"
                    ],

                "url":
                    result.get(
                        "url",
                        ""
                    ),

                "language":
                    result.get(
                        "language"
                    ),

                "metadata":
                    result.get(
                        "metadata",
                        {}
                    ),

            })

        # ----------------------------------------------------------------
        # COLLECTION RESULTS
        # ----------------------------------------------------------------

        for key in (

            "articles",
            "ebooks",
            "books",
            "elibrary",
            "code_books",
            "results",
            "items",

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

                    "text":
                        "\n".join(
                            text_parts
                        ),

                    "url":
                        (
                            item.get(
                                "url"
                            )
                            or item.get(
                                "infoLink",
                                ""
                            )
                        ),

                    "language":
                        item.get(
                            "language"
                        ),

                    "metadata":
                        {
                            "external_record":
                                item
                        },

                })

        # ----------------------------------------------------------------
        # VIDEO RESULTS
        # ----------------------------------------------------------------

        videos = result.get(
            "videos",
            []
        )

        if isinstance(
            videos,
            list,
        ):

            for video in videos:

                if isinstance(
                    video,
                    dict,
                ):

                    documents.extend(
                        self._video_to_documents(
                            video
                        )
                    )

        # ----------------------------------------------------------------
        # DIRECT TRANSCRIPT
        # ----------------------------------------------------------------

        transcript = result.get(
            "transcript"
        )

        if transcript:

            documents.append({

                "text":
                    str(
                        transcript
                    ),

                "url":
                    result.get(
                        "url",
                        ""
                    ),

                "language":
                    result.get(
                        "language"
                    ),

                "metadata":
                    {

                        "content_type":
                            (
                                "video_transcript"
                            ),

                        "transcript":
                            True,

                    },

            })

    # --------------------------------------------------------------------
    # LIST RESPONSE
    # --------------------------------------------------------------------

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

# ========================================================================
# VIDEO NORMALIZATION
# ========================================================================

def _video_to_documents(
    self,
    video: Dict[str, Any],
) -> List[
    Dict[str, Any]
]:
    """
    Convert video information into MemoryGrid documents.
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

        "content_type":
            "video",

        "video_id":
            video.get(
                "id"
            ),

        "channel":
            video.get(
                "channel"
            ),

        "author":
            video.get(
                "author"
            ),

        "published_at":
            video.get(
                "published_at"
            ),

        "duration":
            video.get(
                "duration"
            ),

        "thumbnail":
            video.get(
                "thumbnail"
            ),

    }

    # --------------------------------------------------------------------
    # VIDEO METADATA DOCUMENT
    # --------------------------------------------------------------------

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

            "text":
                metadata_text,

            