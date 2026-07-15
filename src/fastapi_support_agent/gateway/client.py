"""LLM gateway: the single entry point for every model call in this project.

Wraps Groq (primary, free tier) with automatic fallback to Gemini (secondary,
free tier), so a rate-limited or errored primary call doesn't take down the
whole system. No other module should import ChatGroq/ChatGoogleGenerativeAI
directly — everything goes through get_gateway_llm().
"""

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Free-tier caps: Groq llama-3.3-70b-versatile = 30 RPM, Gemini gemini-3.1-flash-lite = 15 RPM.
# Throttle a bit under each so our own limiter kicks in before the provider ever rejects us.
GROQ_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=25 / 60, check_every_n_seconds=0.1, max_bucket_size=4
)
GEMINI_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=12 / 60, check_every_n_seconds=0.1, max_bucket_size=2
)


def get_gateway_llm():
    primary = ChatGroq(model=GROQ_MODEL, temperature=0, rate_limiter=GROQ_RATE_LIMITER)
    fallback = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, temperature=0, rate_limiter=GEMINI_RATE_LIMITER
    )
    return primary.with_fallbacks([fallback])
