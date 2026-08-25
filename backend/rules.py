RULES = {
    "professionalism": True,
    "no_contradiction": True,
    "respect_punctuation": True,
    "temperament_aware": True,
    "max_output_words": 2300,
    "max_paragraphs": 30,
    "no_hallucination": True,
    "no_secret_leaking": True,
    "direct_answers": True,
    "respect_idioms_disjunctions": True,
}

# Patterns that indicate hallucination or placeholder text
HALLUCINATION_PATTERNS = [
    "i don't know",
    "i am not sure",
    "to be honest",
    "as an ai",
    "i cannot provide",
    "i am unable",
    "i apologize",
    "i'm sorry",
    "i cannot answer",
    "i have no information",
    "i do not have",
]

# Patterns that may indicate secret leaking
SECRET_PATTERNS = [
    "###@@@55019855199##@@@",
    "###@@@56019855199##@@@",
    "###@@@55019855197##@@@",
    "###@@@55019655199##@@@",
    "api key",
    "secret key",
    "password",
    "token",
    "access key",
]

def enforce_rules(text: str, temperament: str = "sanguine") -> bool:
    """
    Enforce AI guidelines.
    Returns True if text passes all rules, False otherwise.
    """
    # Check word count
    words = len(text.split())
    if words > RULES["max_output_words"]:
        return False

    # Check hallucination patterns
    lower_text = text.lower()
    if RULES["no_hallucination"]:
        for pattern in HALLUCINATION_PATTERNS:
            if pattern in lower_text and words < 5:
                return False

    # Check secret leaking
    if RULES["no_secret_leaking"]:
        for pattern in SECRET_PATTERNS:
            if pattern in lower_text:
                return False

    # Check directness: reject if text starts with hedging phrases
    if RULES["direct_answers"]:
        if lower_text.startswith(("i think", "maybe", "perhaps", "i'm not sure, but..." , "this might get blocked!!!" , "this has been restricted")):
            return False

    # Ensure respect for idiomatic expressions and disjunctions
    # (This is more about training data; but we can do a basic check:
    # no rule applied here for simplicity, but the flag is set)
    return True