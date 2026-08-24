import os
import httpx
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Existing API functions (dictionary, news, books, elibrary, wikipedia)
# ---------------------------------------------------------------------------

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
    base = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
    if category:
        base += f"&subject={category}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(base)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                ebooks = [
                    {
                        "title": it["volumeInfo"].get("title"),
                        "authors": ", ".join(it["volumeInfo"].get("authors", [])) if it["volumeInfo"].get("authors") else "Unknown",
                        "infoLink": it["volumeInfo"].get("infoLink")
                    }
                    for it in items
                ]
                return {"elibrary": ebooks}
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


# ---------------------------------------------------------------------------
# New API functions
# ---------------------------------------------------------------------------

async def fetch_github_ebooks(query: str) -> dict:
    """
    Search GitHub repositories for ebooks using the GitHub Search API.
    Optional GitHub token for higher rate limits.
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/search/repositories?q={query}+ebook&per_page=5"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                repos = [
                    {
                        "name": r.get("full_name"),
                        "description": r.get("description"),
                        "url": r.get("html_url"),
                        "stars": r.get("stargazers_count")
                    }
                    for r in items
                ]
                return {"ebooks": repos}
            return {"ebooks": [], "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ebooks": [], "error": str(e)}


async def fetch_code_textbook(query: str) -> dict:
    """
    Search for code textbooks using the Google Books API with subject=programming.
    This returns coding books specifically.
    """
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}+programming&maxResults=5"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
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
                return {"code_books": books}
            return {"code_books": []}
        except Exception as e:
            return {"code_books": [], "error": str(e)}


async def fetch_alphavantage(symbol: str) -> dict:
    """
    Fetch stock data from Alpha Vantage API.
    Requires ALPHA_VANTAGE_API_KEY.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"stock": None, "note": "ALPHA_VANTAGE_API_KEY not set"}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {"stock": data.get("Global Quote", {})}
            return {"stock": {}}
        except Exception as e:
            return {"stock": {}, "error": str(e)}


async def fetch_financial_modelling_prep(symbol: str) -> dict:
    """
    Fetch financial statements from Financial Modelling Prep API.
    Requires FMP_API_KEY.
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return {"financial": None, "note": "FMP_API_KEY not set"}
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {"financial": data[0] if data else {}}
            return {"financial": {}}
        except Exception as e:
            return {"financial": {}, "error": str(e)}