"""
Follow-up Questions Module

Generates contextual follow-up question suggestions based on the user's last query
and the AI's last answer. Uses keyword extraction and simple templates.
"""

import re
from intent_analyzer import detect_domain
from symbols import recognize_symbols

STOP_WORDS = {"the","a","an","is","are","was","were","be","been","being","have","has","had",
              "do","does","did","will","would","shall","should","may","might","must","can","could",
              "of","in","on","at","by","for","with","about","against","between","into","through",
              "during","before","after","above","below","to","from","up","down","out","off","over",
              "under","again","further","then","once","here","there","when","where","why","how",
              "all","any","both","each","few","more","most","other","some","such","no","nor","not",
              "only","own","same","so","than","too","very","s","t","just","don","now","i","me","my",
              "we","our","ours","you","your","yours","he","him","his","she","her","hers","it","its",
              "they","them","their","theirs","what","which","who","whom","this","that","these","those"}

def extract_keywords(text: str, top_n=5):
    """Extract top keywords excluding stop words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w,_ in sorted_words[:top_n]]

def generate_follow_ups(query: str, answer: str, domain: str = "general", max_questions=3):
    """Generate follow-up questions using keywords and domain templates."""
    keywords = extract_keywords(query)
    if not keywords:
        keywords = extract_keywords(answer)

    questions = []
    if domain == "code":
        if keywords:
            questions.append(f"Can you show me an example of using {keywords[0]}?")
            questions.append(f"What are common errors with {keywords[0]}?")
            questions.append(f"How does {keywords[0]} relate to {keywords[1] if len(keywords)>1 else 'other concepts'}?")
    elif domain == "medical":
        if keywords:
            questions.append(f"What are the symptoms of {keywords[0]}?")
            questions.append(f"What treatments are available for {keywords[0]}?")
            questions.append(f"Are there any side effects of {keywords[0]}?")
    elif domain == "business":
        if keywords:
            questions.append(f"How does {keywords[0]} impact small businesses?")
            questions.append(f"What are the market trends for {keywords[0]}?")
            questions.append(f"Can you explain {keywords[0]} with an example?")
    else:
        if keywords:
            questions.append(f"Can you tell me more about {keywords[0]}?")
            questions.append(f"How does {keywords[0]} work?")
            questions.append(f"What are the key aspects of {keywords[0]}?")

    # Ensure unique and limit
    unique_questions = list(dict.fromkeys(questions))
    return unique_questions[:max_questions]

def extract_word_pairs(text: str):
    """
    Extract adjacent word pairs from text.

    Example:
        "python is a programming language"

    produces:
        python_is
        is_a
        a_programming
        programming_language
    """
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    return [
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    ]


def extract_next_word_candidates(text: str, limit=5):
    """
    Find likely next words from observed adjacent word sequences.

    This is intentionally lightweight and deterministic.
    It does not require embeddings or vector databases.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    if len(words) < 2:
        return []

    transitions = defaultdict(Counter)

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]

        if current_word in STOP_WORDS:
            continue

        transitions[current_word][next_word] += 1

    candidates = []

    for word, next_words in transitions.items():
        for next_word, count in next_words.most_common(limit):
            candidates.append({
                "word": next_word,
                "after": word,
                "count": count,
                "pair": f"{word}_{next_word}",
            })

    candidates.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return candidates[:limit]