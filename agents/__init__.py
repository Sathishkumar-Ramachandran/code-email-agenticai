# agents package


def extract_text_content(content) -> str:
    """Safely extract text from an LLM response's .content attribute.

    With newer Gemini / langchain-google-genai versions the content can be:
      - a plain str
      - a list of dicts like [{"type": "text", "text": "..."}]
      - a list of strings
    This helper normalises all variants to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
