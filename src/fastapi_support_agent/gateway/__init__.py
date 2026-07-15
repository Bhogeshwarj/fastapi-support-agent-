"""Single entry point for the LLM gateway: gateway_invoke().

Every future node/tool that needs a model response should call gateway_invoke()
rather than constructing ChatGroq/ChatGoogleGenerativeAI or calling
get_gateway_llm()/invoke_with_tracking() separately.
"""

from .client import get_gateway_llm
from .tracking import invoke_with_tracking

_llm = None


def gateway_invoke(messages):
    global _llm
    if _llm is None:
        _llm = get_gateway_llm()
    return invoke_with_tracking(_llm, messages)
