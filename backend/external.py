"""
CoMpaNeoN External Source Adapter Layer
========================================

External acquisition providers used by web_crawler.py.

Responsibilities
----------------
- Define external source directions/endpoints.
- Provide normalized adapters for external APIs.
- Support dictionaries, news, books, e-library, Wikipedia,
  GitHub, financial sources, APITube and YouTube.
- Return normalized external documents to web_crawler.py.
- Preserve source identity and provenance.
- Detect language where possible.
- Provide video metadata, subtitles and transcript information
  where legitimately available.

IMPORTANT
---------
external.py does NOT:

- crawl autonomously
- decide MemoryGrid placement
- perform GSP placement
- rank knowledge
- build WordChain relationships
- perform WordUnderstanding
- manage STM/LTM
- generate prompts
- generate AI responses

Architecture
------------

                external.py
                     │
             source adapters
                     │
                     ▼
               web_crawler.py
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   validate       checksum      normalize
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                MemoryGrid
                     │
                     ▼
                  Ranking
                     │
                     ▼
                 WordChain
                     │
                     ▼
              WordUnderstanding

The crawler is the owner of acquisition.
This module simply knows how to communicate with external sources.
"""

from __future__ import annotations

import hashlib
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import httpx

try:
    from langdetect import detect as detect_language
except ImportError:
    detect_language = None


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TIMEOUT = 20.0
DEFAULT_LIMIT = 5


# ============================================================================
# SOURCE REGISTRY
# ============================================================================

EXTERNAL_SOURCES: Dict[str, Dict[str, Any]] = {

    "dictionary": {
        "type": "dictionary",
        "provider": "dictionaryapi.dev",
        "base_url": (
            "https://api.dictionaryapi.dev/api/v2/entries/en"
        ),
        "requires_key": False,
        "capabilities": [
            "definitions",
            "meanings",
            "phonetics",
        ],
    },

    "news": {
        "type": "news",
        "provider": "gnews",
        "base_url": (
            "https://gnews.io/api/v4/search"
        ),
        "requires_key": True,
        "environment_key": "GNEWS_API_KEY",
        "capabilities": [
            "search",
            "articles",
            "publication_date",
        ],
    },

    "openlibrary": {
        "type": "book",
        "provider": "openlibrary",
        "base_url": (
            "https://openlibrary.org/search.json"
        ),
        "requires_key": False,
        "capabilities": [
            "books",
            "authors",
            "subjects",
            "publication_year",
        ],
    },

    "google_books": {
        "type": "ebook",
        "provider": "google_books",
        "base_url": (
            "https://www.googleapis.com/books/v1/volumes"
        ),
        "requires_key": False,
        "capabilities": [
            "books",
            "authors",
            "subjects",
            "metadata",
            "preview",
        ],
    },

    "wikipedia": {
        "type": "encyclopedia",
        "provider": "wikipedia",
        "base_url": (
            "https://en.wikipedia.org/w/api.php"
        ),
        "requires_key": False,
        "capabilities": [
            "search",
            "summary",
            "encyclopedia",
        ],
    },

    "github": {
        "type": "repository",
        "provider": "github",
        "base_url": (
            "https://api.github.com"
        ),
        "requires_key": False,
        "environment_key": "GITHUB_TOKEN",
        "capabilities": [
            "repositories",
            "code",
            "documentation",
            "ebooks",
        ],
    },

    "alphavantage": {
        "type": "financial",
        "provider": "alpha_vantage",
        "base_url": (
            "https://www.alphavantage.co/query"
        ),
        "requires_key": True,
        "environment_key": "ALPHA_VANTAGE_API_KEY",
        "capabilities": [
            "stocks",
            "market_data",
        ],
    },

    "financial_modeling_prep": {
        "type": "financial",
        "provider": "financial_modeling_prep",
        "base_url": (
            "https://financialmodelingprep.com/api/v3"
        ),
        "requires_key": True,
        "environment_key": "FMP_API_KEY",
        "capabilities": [
            "company_profile",
            "financial_data",
        ],
    },

    "apitube": {
        "type": "video",
        "provider": "apitube",
        "base_url": (
            "https://api.apitube.io"
        ),
        "requires_key": True,
        "environment_key": "APITUBE_API_KEY",
        "capabilities": [
            "video_search",
            "video_metadata",
            "media_discovery",
        ],
    },

    "youtube": {
        "type": "video",
        "provider": "youtube",
        "base_url": (
            "https://www.googleapis.com/youtube/v3"
        ),
        "requires_key": True,
        "environment_key": "YOUTUBE_API_KEY",
        "capabilities": [
            "video_search",
            "video_metadata",
            "channel_metadata",
            "captions",
        ],
    },
}


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_source(
    source: str,
) -> str:
    """
    Normalize an external source name.
    """

    return str(
        source or ""
    ).strip().lower()


def get_source(
    source: str,
) -> Optional[Dict[str, Any]]:
    """
    Return source configuration.
    """

    return EXTERNAL_SOURCES.get(
        normalize_source(source)
    )


def list_sources() -> List[str]:
    """
    Return all registered external sources.
    """

    return list(
        EXTERNAL_SOURCES.keys()
    )


# ============================================================================
# LANGUAGE
# ============================================================================

def detect_content_language(
    text: str,
) -> Optional[str]:
    """
    Detect language using langdetect when available.

    This is deliberately advisory.

    tokenizer.py remains the canonical linguistic tokenizer.
    """

    if not text:
        return None

    if detect_language is None:
        return None

    try:
        return detect_language(
            str(text)
        )
    except Exception:
        return None


# ============================================================================
# CHECKSUM
# ============================================================================

def content_checksum(
    content: str,
) -> str:
    """
    Produce a deterministic SHA-256 checksum.

    The crawler can use this to identify duplicate content.
    """

    return hashlib.sha256(
        str(content).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================================
# NORMALIZED DOCUMENT
# ============================================================================

def normalize_document(
    *,
    source: str,
    source_type: str,
    content: str = "",
    title: str = "",
    url: str = "",
    external_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert an external result into the common crawler document format.

    web_crawler.py consumes this structure.
    """

    metadata = dict(
        metadata or {}
    )

    language = (
        metadata.get("language")
        or detect_content_language(
            content
        )
    )

    return {
        "source": normalize_source(
            source
        ),

        "source_type": source_type,

        "external_id": external_id,

        "url": url,

        "title": title,

        "content": content,

        "language": language,

        "checksum": content_checksum(
            content
        ) if content else None,

        "metadata": metadata,
    }


# ============================================================================
# HTTP CLIENT
# ============================================================================

def _client() -> httpx.AsyncClient:
    """
    Construct a standard external HTTP client.

    The crawler controls when requests are made.
    """

    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "CoMpaNeoN-WebCrawler/1.0"
            )
        },
    )


# ============================================================================
# DICTIONARY
# ============================================================================

async def fetch_dictionary(
    word: str,
) -> Dict[str, Any]:

    source = get_source(
        "dictionary"
    )

    if not source:
        return {
            "documents": []
        }

    url = (
        f"{source['base_url']}/"
        f"{word}"
    )

    async with _client() as client:

        try:

            response = await client.get(
                url
            )

            if response.status_code != 200:
                return {
                    "documents": []
                }

            data = response.json()

            if not data:
                return {
                    "documents": []
                }

            entry = data[0]

            meanings = entry.get(
                "meanings",
                []
            )

            content = (
                f"{word}\n"
                f"{meanings}"
            )

            document = normalize_document(
                source="dictionary",
                source_type="dictionary",
                content=content,
                title=word,
                url=url,
                external_id=word,
                metadata={
                    "meanings": meanings,
                    "phonetics": entry.get(
                        "phonetics",
                        []
                    ),
                },
            )

            return {
                "documents": [
                    document
                ]
            }

        except Exception as exc:

            return {
                "documents": [],
                "error": str(exc),
            }


# ============================================================================
# NEWS
# ============================================================================

async def fetch_news(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:

    api_key = os.getenv(
        "GNEWS_API_KEY"
    )

    if not api_key:
        return {
            "documents": [],
            "note": (
                "GNEWS_API_KEY not set"
            ),
        }

    source = get_source(
        "news"
    )

    params = {
        "q": query,
        "token": api_key,
        "lang": "en",
        "max": limit,
    }

    async with _client() as client:

        try:

            response = await client.get(
                source["base_url"],
                params=params,
            )

            if response.status_code != 200:
                return {
                    "documents": []
                }

            articles = response.json().get(
                "articles",
                []
            )

            documents = []

            for article in articles:

                content = (
                    article.get(
                        "content"
                    )
                    or article.get(
                        "description"
                    )
                    or ""
                )

                documents.append(
                    normalize_document(
                        source="news",
                        source_type="article",
                        content=content,
                        title=article.get(
                            "title",
                            "",
                        ),
                        url=article.get(
                            "url",
                            "",
                        ),
                        external_id=article.get(
                            "url"
                        ),
                        metadata={
                            "published_at": article.get(
                                "publishedAt"
                            ),
                            "source_name": (
                                article.get(
                                    "source",
                                    {}
                                ).get(
                                    "name"
                                )
                            ),
                            "image": article.get(
                                "image"
                            ),
                        },
                    )
                )

            return {
                "documents": documents
            }

        except Exception as exc:

            return {
                "documents": [],
                "error": str(exc),
            }


# ============================================================================
# OPENLIBRARY
# ============================================================================

async def fetch_books(
    query: str,
    category: str = "",
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:

    source = get_source(
        "openlibrary"
    )

    params = {
        "q": query,
        "limit": limit,
    }

    if category:
        params[
            "subject"
        ] = category

    async with _client() as client:

        try:

            response = await client.get(
                source["base_url"],
                params=params,
            )

            if response.status_code != 200:
                return {
                    "documents": []
                }

            docs = response.json().get(
                "docs",
                []
            )

            documents = []

            for book in docs:

                title = book.get(
                    "title",
                    ""
                )

                authors = book.get(
                    "author_name",
                    []
                )

                content = (
                    f"Title: {title}\n"
                    f"Authors: "
                    f"{', '.join(authors)}\n"
                    f"Subjects: "
                    f"{', '.join(book.get('subject', [])[:10])}"
                )

                documents.append(
                    normalize_document(
                        source="openlibrary",
                        source_type="book",
                        content=content,
                        title=title,
                        external_id=str(
                            book.get(
                                "key",
                                title,
                            )
                        ),
                        metadata={
                            "authors": authors,
                            "year": book.get(
                                "first_publish_year"
                            ),
                            "subjects": book.get(
                                "subject",
                                []
                            )[:20],
                        },
                    )
                )

            return {
                "documents": documents
            }

        except Exception as exc:

            return {
                "documents": [],
                "error": str(exc),
            }


# ============================================================================
# GOOGLE BOOKS / E-LIBRARY
# ============================================================================

async def fetch_elibrary(
    query: str,
    category: str = "",
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:

    source = get_source(
        "google_books"
    )

    params = {
        "q": query,
        "maxResults": limit,
    }

    if category:
        params[
            "subject"
        ] = category

    async with _client() as client:

        try:

            response = await client.get(
                source["base_url"],
                params=params,
            )

            if response.status_code != 200:
                return {
                    "documents": []
                }

            items = response.json().get(
                "items",
                []
            )

            documents = []

            for item in items:

                info = item.get(
                    "volumeInfo",
                    {}
                )

                title = info.get(
                    "title",
                    ""
                )

                authors = info.get(
                    "authors",
                    []
                )

                description = info.get(
                    "description",
                    ""
                )

                content = (
                    f"Title: {title}\n"
                    f"Authors: "
                    f"{', '.join(authors)}\n"
                    f"{description}"
                )

                documents.append(
                    normalize_document(
                        source="google_books",
                        source_type="ebook",
                        content=content,
                        title=title,
                        url=info.get(
                            "infoLink",
                            "",
                        ),
                        external_id=item.get(
                            "id"
                        ),
                        metadata={
                            "authors": authors,
                            "published_date": info.get(
                                "publishedDate"
                            ),
                            "categories": info.get(
                                "categories",
                                []
                            ),
                            "description": description,
                        },
                    )
                )

            return {
                "documents": documents
            }

        except Exception as exc:

            return {
                "documents": [],
                "error": str(exc),
            }


# ============================================================================
# WIKIPEDIA
# ============================================================================

async def fetch_wikipedia(
    query: str,
    lang: str = "en",
    limit: int = 3,
) -> Dict[str, Any]:

    search_url = (
        f"https://{lang}.wikipedia.org/"
        "w/api.php"
    )

    summary_base = (
        f"https://{lang}.wikipedia.org/"
        "api/rest_v1/page/summary/"
    )

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "srprop": "",
    }

    async with _client() as client:

        try:

            response = await client.get(
                sear