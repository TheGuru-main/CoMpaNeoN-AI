import httpx
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class WebCrawler:
    def __init__(self):
        self.word_index = {}  # 26x1 grid: col -> list of words

    def fetch_text(self, url: str) -> str:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def index_words(self, text: str):
        words = text.split()
        for w in words:
            if not w:
                continue
            col = ord(w[0].lower()) - 97 if 'a' <= w[0].lower() <= 'z' else 0
            self.word_index.setdefault(col, []).append(w.lower())

    def crawl(self, url: str) -> str:
        text = self.fetch_text(url)
        self.index_words(text)
        return text