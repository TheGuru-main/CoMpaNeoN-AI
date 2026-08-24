def build_prompt(query, context, last_message="", lang="en"):
    prompt = f"User is speaking {lang}.\n"
    if last_message:
        prompt += f"Last message: {last_message}\n"
    prompt += f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
    return prompt