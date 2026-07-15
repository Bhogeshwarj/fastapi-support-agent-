"""Usage/cost tracking for every gateway call.

Wraps LangChain's UsageMetadataCallbackHandler, which reads the standardized
usage_metadata field every chat model now returns, to log token counts, which
provider actually served each call, and an estimated cost. Both providers are
free-tier today, so real cost is $0 - these are published paid-tier rates,
tracked for capacity planning before this ever needs to become a paid setup.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler

LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "usage_log.jsonl"

# USD per 1M tokens, as (input_rate, output_rate). Both providers are on free
# tiers for this project, so nothing is actually billed - these are published
# paid-tier rates, kept only so estimated cost is visible if that ever changes.
PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def invoke_with_tracking(llm, messages):
    callback = UsageMetadataCallbackHandler()
    response = llm.invoke(messages, config={"callbacks": [callback]})

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        for model_name, usage in callback.usage_metadata.items():
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            in_rate, out_rate = PRICING.get(model_name, (0.0, 0.0))
            cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost_usd": round(cost, 6),
                    }
                )
                + "\n"
            )

    return response
