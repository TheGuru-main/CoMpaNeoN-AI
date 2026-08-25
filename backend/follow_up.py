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

def generate_follow_ups(
    query: str,
    answer: str,
    domain: str = "general",
    max_questions=3
):
    """
    Generate contextual follow-up questions.

    Uses:
    - query keywords
    - answer keywords
    - word-pair continuity
    - domain
    - previous conversational subject
    """

    query_keywords = extract_keywords(query)
    answer_keywords = extract_keywords(answer)

    keywords = query_keywords or answer_keywords

    if not keywords:
        return []

    questions = []

    primary = keywords[0]
    secondary = keywords[1] if len(keywords) > 1 else None

    # ---------------------------------------------------------
    # CONTEXTUAL QUESTIONS
    # ---------------------------------------------------------

    if domain == "code":
        questions.extend([
            f"Can you show me an example of using {primary}?",
            f"What are the common errors when using {primary}?",
        ])

        if secondary:
            questions.append(
                f"How does {primary} relate to {secondary}?"
            )

    elif domain == "medical":
        questions.extend([
            f"What are the symptoms associated with {primary}?",
            f"What treatments are commonly used for {primary}?",
            f"What should someone know before considering treatment for {primary}?",
        ])

    elif domain == "business":
        questions.extend([
            f"How does {primary} affect a business?",
            f"What are the main considerations when dealing with {primary}?",
            f"Can you give me a practical example involving {primary}?",
        ])

    else:
        questions.extend([
            f"Can you explain {primary} in more detail?",
            f"How does {primary} work?",
            f"What is the relationship between {primary} and {secondary}?"
            if secondary
            else f"What are the key aspects of {primary}?",
        ])

    # ---------------------------------------------------------
    # WORD-PAIR CONTINUITY
    # ---------------------------------------------------------

    pairs = extract_word_pairs(query)

    if pairs:
        last_pair = pairs[-1]

        if "_" in last_pair:
            first, second = last_pair.split("_", 1)

            questions.append(
                f"How does {first} relate to {second}?"
            )

    # ---------------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------------

    unique_questions = []

    for question in questions:
        question = question.strip()

        if question and question not in unique_questions:
            unique_questions.append(question)

    return unique_questions[:max_questions]