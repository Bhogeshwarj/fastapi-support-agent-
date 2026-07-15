"""Manual smoke test for the gateway client. Run: uv run scripts/test_gateway.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi_support_agent.gateway.client import get_gateway_llm

if __name__ == "__main__":
    llm = get_gateway_llm()
    response = llm.invoke("In one sentence, what is FastAPI?")
    print("Response:", response.content)
    print("Served by:", response.response_metadata.get("model_name", "unknown"))
