import httpx
from bs4 import BeautifulSoup
from page_cache import PageCache
from crawler_scheduler import CrawlerScheduler, ContentType

class WebCrawler:
    def __init__(self):
        self.word_index = {}
        self.page_cache = PageCache()
        self.scheduler = CrawlerScheduler()

    def fetch_text(self, url):
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def index_words(self, text):
        for w in text.split():
            if not w:
                continue
            col = ord(w[0].lower()) - 97 if 'a' <= w[0].lower() <= 'z' else 0
            self.word_index.setdefault(col, []).append(w.lower())

    def crawl(self, url, source_type=ContentType.NORMAL_WEB):
        # Check cache first
        cached_data, cached_hash = self.page_cache.get(url)
        if cached_data is not None:
            return cached_data

        text = self.fetch_text(url)
        content_hash = self.page_cache._hash(text)
        if self.page_cache.has_changed(url, text):
            # Process (index words, etc.)
            self.index_words(text)
            # Store in cache
            self.page_cache.set(url, content_hash, text)
        return text