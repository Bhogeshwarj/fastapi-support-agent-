"""LLM gateway: the single entry point for every model call in this project.

Wraps Groq (primary, free tier) with automatic fallback to Gemini (secondary,
free tier), so a rate-limited or errored primary call doesn't take down the
whole system. No other module should import ChatGroq/ChatGoogleGenerativeAI
directly — everything goes through get_gateway_llm().
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.1-flash-lite"


def get_gateway_llm():
    primary = ChatGroq(model=GROQ_MODEL, temperature=0)
    fallback = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    return primary.with_fallbacks([fallback])
