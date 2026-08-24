"""
Crawler scheduling with frequency based on content type.
Queue with priority, frequency, and content hash.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
from enum import Enum

class ContentType(Enum):
    HOT_NEWS = 1
    NORMAL_WEB = 2
    LOW_CHANGE = 3

FREQUENCIES = {
    ContentType.HOT_NEWS: timedelta(minutes=30),
    ContentType.NORMAL_WEB: timedelta(days=2),
    ContentType.LOW_CHANGE: timedelta(weeks=2),
}

@dataclass
class CrawlQueueItem:
    url: str
    priority: int = 5
    last_crawled: datetime = None
    next_crawl: datetime = None
    frequency: timedelta = None
    content_hash: str = None
    status: str = "pending"
    depth: int = 0
    source_type: ContentType = ContentType.NORMAL_WEB

class CrawlerScheduler:
    def __init__(self):
        self.queue = []

    def add_url(self, url, source_type=ContentType.NORMAL_WEB, depth=0):
        item = CrawlQueueItem(
            url=url,
            source_type=source_type,
            depth=depth,
            frequency=FREQUENCIES[source_type],
        )
        self.queue.append(item)
        self.queue.sort(key=lambda x: (x.next_crawl or datetime.min, x.priority))

    def get_next_due(self):
        now = datetime.utcnow()
        for item in self.queue:
            if item.status == "pending" and (item.next_crawl is None or now >= item.next_crawl):
                return item
        return None

    def mark_crawled(self, url, content_hash, next_crawl=None):
        for item in self.queue:
            if item.url == url:
                item.content_hash = content_hash
                item.last_crawled = datetime.utcnow()
                item.next_crawl = next_crawl or datetime.utcnow() + item.frequency
                item.status = "scheduled"
                break

    def remove(self, url):
        self.queue = [i for i in self.queue if i.url != url]