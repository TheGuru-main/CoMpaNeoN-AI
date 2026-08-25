DOMAIN_KEYWORDS = {
    "code": ["def", "class", "import", "function", "return", "if", "else", "loop", "variable", "algorithm", "debug", "compiler", "library", "api", "database", "framework", "python", "javascript", "java", "c++", "c#", "ruby", "php", "sql", "html", "css", "python_rules", "fix_my_code", "debug"],
    "medical": ["diagnosis","treatment","symptom","surgery","disease","patient","clinical","pharmacy","drug"],
    "legal": ["law","court","attorney","legal","statute","regulation","compliance","lawsuit","judge"],
    "education": ["homework","explain","definition","learn","teach","curriculum","classroom","student","school"],
    "sports": ["football","basketball","tennis","goal","match","tournament","player","league","championship"],
    "business": ["stock","revenue","profit","market","economy","investment","trade","finance","entrepreneur"],
    "agriculture": ["crop","farm","harvest","soil","livestock","fertilizer","agriculture","irrigation","poultry"],
    "technology": ["code","software","ai","blockchain","server","api","programming","app","cloud"],
    "religious": ["prayer","quran","bible","allah","god","surah","zakat","church","mosque"],
    "news": ["breaking","headlines","latest","report","update","today","journal","news"]
}

def detect_domain(query: str) -> str:
    q = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return domain
    return "general"