"""Manual smoke test for the gateway. Run: uv run scripts/test_gateway.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi_support_agent.gateway import gateway_invoke
from fastapi_support_agent.gateway.tracking import LOG_PATH

if __name__ == "__main__":
    response = gateway_invoke("In one sentence, what is FastAPI?")
    print("Response:", response.content)
    print("Served by:", response.response_metadata.get("model_name", "unknown"))
    print()
    print(f"Usage log ({LOG_PATH}):")
    print(LOG_PATH.read_text().strip().splitlines()[-1])
