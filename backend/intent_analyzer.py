import re

DOMAIN_KEYWORDS = {
   "code": ["def", "class", "import", "function", "return", "if", "else", "loop", "variable", "algorithm", "debug", "compiler", "library", "API", "database", "framework", "python", "javascript", "java", "C++", "C#", "ruby", "php", "sql", "html", "css"],
    "medical":     ["diagnosis","treatment","symptom","surgery","disease","patient","clinical","pharmacy","drug"],
    "legal":       ["law","court","attorney","legal","statute","regulation","compliance","lawsuit","judge"],
    "education":   ["homework","explain","definition","learn","teach","curriculum","classroom","student","school"],
    "sports":      ["football","basketball","tennis","goal","match","tournament","player","league","championship"],
    "business":    ["stock","revenue","profit","market","economy","investment","trade","finance","entrepreneur"],
    "agriculture": ["crop","farm","harvest","soil","livestock","fertilizer","agriculture","irrigation","poultry"],
    "technology":  ["code","software","AI","blockchain","server","API","programming","app","cloud"],
    "religious":   ["prayer","quran","bible","allah","god","surah","zakat","church","mosque"],
    "news":        ["breaking","headlines","latest","report","update","today","journal","news"]
}

def detect_domain(query: str) -> str:
    """Return the best‑matching domain for the given query, or 'general'."""
    q = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return domain
    return "general"

def analyze_intent(query: str) -> dict:
    """Return domain and basic intent for the query."""
    return {
        "domain": detect_domain(query),
        "query": query
    }