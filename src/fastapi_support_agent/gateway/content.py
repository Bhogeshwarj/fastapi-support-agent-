"""Normalize chat model response content across providers.

Groq returns response.content as a plain string. Gemini 3.x returns a list of
content blocks instead (e.g. [{"type": "text", "text": "...", "extras": {...}}]),
seen repeatedly once the gateway falls back to Gemini. Anything that reads a
final answer should go through this instead of assuming .content is a string.
"""


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)
