"""Search FastAPI's GitHub issue tracker live via the REST Search API.

Unlike docs/changelog, issues are never bulk-fetched - freshness (open/closed
state, recent comments) matters more than a static snapshot, so this queries
GitHub's Search API at ask-time using a fine-grained, public-repo-read-only
token (GITHUB_TOKEN in .env). Falls back to unauthenticated requests (much
lower rate limit) if no token is set.
"""

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
REPO = "fastapi/fastapi"
API_VERSION = "2026-03-10"


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": API_VERSION}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@tool
def search_github_issues(query: str, state: str = "all", max_results: int = 5) -> str:
    """Search FastAPI's GitHub issues for reports matching a query.

    Use this to check if a problem/error/behavior has already been reported,
    find workarounds, or check if something is a known bug. `state` can be
    "open", "closed", or "all" (default).
    """
    search_query = f"repo:{REPO} is:issue {query}"
    if state in ("open", "closed"):
        search_query += f" state:{state}"

    response = requests.get(
        f"{GITHUB_API_BASE}/search/issues",
        headers=_headers(),
        params={"q": search_query, "sort": "updated", "order": "desc", "per_page": max_results},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    items = data.get("items", [])
    if not items:
        return f"No GitHub issues found matching '{query}'."

    lines = [
        f"Found {data.get('total_count', len(items))} issue(s) matching '{query}' "
        f"(showing top {len(items)}):"
    ]
    for issue in items:
        lines.append(
            f"- #{issue['number']} [{issue['state']}] {issue['title']} - {issue['html_url']} "
            f"(updated {issue['updated_at'][:10]})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(search_github_issues.invoke({"query": "lifespan startup"}))
