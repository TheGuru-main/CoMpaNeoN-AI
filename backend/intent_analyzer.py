import re

DOMAIN_KEYWORDS = {
    "medical": ["diagnosis","treatment","symptom","surgery"],
    "legal": ["law","court","attorney","legal"],
    "education": ["homework","explain","definition","learn"],
    "technology": ["code","software","AI","blockchain","server"],
    "business": ["stock","revenue","profit","market"],
}
def detect_domain(text):
    t = text.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return domain
    return "general"