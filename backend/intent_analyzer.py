import re

DOMAIN_KEYWORDS = {
    "medical": ["diagnosis", "treatment", "symptom", "surgery", "disease", "patient", "clinical"],
    "legal": ["law", "court", "attorney", "legal", "statute", "regulation", "compliance"],
    "education": ["homework", "explain", "definition", "learn", "teach", "curriculum", "student"],
    "sports": ["football", "basketball", "tennis", "goal", "match", "tournament", "player"],
    "business": ["stock", "revenue", "profit", "market", "economy", "investment", "trade"],
    "technology": ["code", "software", "AI", "blockchain", "server", "API", "programming", "app"],
    "agriculture": ["crop", "farm", "harvest", "soil", "livestock", "fertilizer", "irrigation"],
    "religious": ["prayer", "quran", "bible", "allah", "god", "surah", "zakat"],
    "news": ["breaking", "headlines", "latest", "report", "update", "today", "journal"],
}

WORKSPACE_KEYWORDS = {
    "search": [],
    "social": ["friend", "follow", "message", "chat", "profile", "post", "comment", "group"],
    "edu": ["classroom", "assignment", "lesson", "teacher", "student", "exam", "waec", "jamb"],
    "games": ["ludo", "chess", "draft", "snooker", "game", "play", "match", "tournament"],
    "shop": ["buy", "shop", "market", "price", "order", "delivery", "merchant", "product"],
    "code": ["function", "class", "import", "def", "return", "if", "else", "loop", "variable", "algorithm", "debug", "compiler", "library", "API", "database", "framework"]
}

def detect_domain(text):
    t = text.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return domain
    return "general"

def detect_workspace(text):
    t = text.lower()
    for ws, kws in WORKSPACE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return ws
    return "search"

def analyze_intent(text):
    domain = detect_domain(text)
    workspace = detect_workspace(text)
    return {
        "domain": domain,
        "workspace": workspace,
        "query": text
    }