import os
import httpx
from typing import Optional, List, Dict, Any

async def fetch_dictionary(word: str) -> dict:
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return {"definitions": resp.json()[0].get("meanings", [])}
            return {"definitions": []}
        except Exception as e:
            return {"definitions": [], "error": str(e)}

async def fetch_news(query: str) -> dict:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return {"articles": [], "note": "GNEWS_API_KEY not set"}
    url = f"https://gnews.io/api/v4/search?q={query}&token={api_key}&lang=en&max=5"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return {"articles": resp.json().get("articles", [])}
            return {"articles": []}
        except Exception as e:
            return {"articles": [], "error": str(e)}

async def fetch_books(query: str, category: str = "") -> dict:
    """Fetch books from Open Library, optionally filtered by subject."""
    base = f"https://openlibrary.org/search.json?q={query}&limit=5"
    if category:
        base += f"&subject={category}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(base)
            if resp.status_code == 200:
                docs = resp.json().get("docs", [])
                books = [
                    {
                        "title": d.get("title"),
                        "author": ", ".join(d.get("author_name", [])) if d.get("author_name") else "Unknown",
                        "year": d.get("first_publish_year")
                    }
                    for d in docs
                ]
                return {"books": books}
            return {"books": []}
        except Exception as e:
            return {"books": [], "error": str(e)}

async def fetch_elibrary(query: str, category: str = "") -> dict:
    """Fetch from Google Books, optionally filtered by category."""
    base = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
    if category:
        base += f"&subject={category}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(base)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                books = [
                    {
                        "title": it["volumeInfo"].get("title"),
                        "authors": ", ".join(it["volumeInfo"].get("authors", [])) if it["volumeInfo"].get("authors") else "Unknown",
                        "infoLink": it["volumeInfo"].get("infoLink")
                    }
                    for it in items
                ]
                return {"elibrary": books}
            return {"elibrary": []}
        except Exception as e:
            return {"elibrary": [], "error": str(e)}

async def fetch_wikipedia(query: str, lang: str = "en", limit: int = 3) -> dict:
    WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "srprop": "",
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(WIKI_SEARCH_URL, params=params)
            if resp.status_code != 200:
                return {"articles": []}
            data = resp.json()
            titles = [r["title"] for r in data.get("query", {}).get("search", [])]
            articles = []
            for title in titles:
                try:
                    sum_resp = await client.get(f"{WIKI_SUMMARY_URL}{title}")
                    if sum_resp.status_code == 200:
                        sdata = sum_resp.json()
                        articles.append({
                            "title": title,
                            "extract": sdata.get("extract", "")[:500],
                            "url": sdata.get("content_urls", {}).get("desktop", {}).get("page", "")
                        })
                except Exception:
                    continue
            return {"articles": articles}
        except Exception as e:
            return {"articles": [], "error": str(e)}