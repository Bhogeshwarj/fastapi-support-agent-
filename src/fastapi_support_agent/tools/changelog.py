"""Parse FastAPI's release-notes.md into structured entries.

Backs the version-lookup and "is X deprecated" tools. Deliberately separate
from the RAG corpus (rag/chunking.py excludes this file) - this needs exact,
structured lookups by version/keyword, not semantic search over prose.

Formatting note: older entries (pre-~2023) use "## Category" (H2) for
category headers directly under the version header, while newer entries use
"### Category" (H3). Both are handled - category detection just needs to not
be a version header.

Known limitation: very old entries (roughly FastAPI's earliest versions)
predate the "PR [#N](url) by [@author]" attribution convention and use a
plain bullet format instead - those are not captured. Verified against the
real file: 12 pre-PR-era/prose mentions of "deprecat" are missed out of 28
total substring matches; all are from historical versions, not anything a
current support question would realistically ask about.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

CHANGELOG_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "fastapi-docs"
    / "docs"
    / "en"
    / "docs"
    / "release-notes.md"
)

VERSION_HEADER_RE = re.compile(r"^##\s+(?P<version>\d+\.\d+\.\d+)\s*\((?P<date>[\d-]+)\)\s*$")
BULLET_RE = re.compile(
    r"^\*\s*\S*\s*(?P<description>.+?)\.\s*PR\s*\[#(?P<pr_number>\d+)\]\((?P<pr_url>[^)]+)\)"
    r"\s*by\s*\[@(?P<author>[^\]]+)\]"
)


@dataclass
class ChangelogEntry:
    version: str
    date: str | None
    category: str
    description: str
    pr_number: str
    pr_url: str


def parse_changelog() -> list[ChangelogEntry]:
    lines = CHANGELOG_PATH.read_text(encoding="utf-8").splitlines()

    entries: list[ChangelogEntry] = []
    current_version: str | None = None
    current_date: str | None = None
    current_category = "Uncategorized"

    for line in lines:
        version_match = VERSION_HEADER_RE.match(line)
        if version_match:
            current_version = version_match.group("version")
            current_date = version_match.group("date")
            current_category = "Uncategorized"
            continue

        if line.strip() == "## Latest Changes":
            current_version = "unreleased"
            current_date = None
            current_category = "Uncategorized"
            continue

        if line.startswith("## ") or line.startswith("### "):
            current_category = line.lstrip("#").strip()
            continue

        bullet_match = BULLET_RE.match(line)
        if bullet_match and current_version:
            entries.append(
                ChangelogEntry(
                    version=current_version,
                    date=current_date,
                    category=current_category,
                    description=bullet_match.group("description").strip(),
                    pr_number=bullet_match.group("pr_number"),
                    pr_url=bullet_match.group("pr_url"),
                )
            )

    return entries


def _search(term: str, entries: list[ChangelogEntry]) -> list[ChangelogEntry]:
    term_lower = term.lower()
    matches = [e for e in entries if term_lower in e.description.lower()]
    # Sort most recent first. Unreleased entries have no date - treat as newest.
    return sorted(matches, key=lambda e: e.date or "9999-99-99", reverse=True)


_COLD_START_MESSAGE = (
    "The documentation/changelog index is still being built after a cold "
    "start - this takes a minute or two. Please retry shortly."
)


@tool
def lookup_changelog_version(version: str) -> str:
    """Look up what changed in a specific FastAPI version, e.g. "0.139.0".

    Returns every changelog entry recorded for that exact version number.
    """
    if not CHANGELOG_PATH.exists():
        return _COLD_START_MESSAGE
    entries = [e for e in parse_changelog() if e.version == version]
    if not entries:
        return f"No changelog entries found for version {version}."
    lines = [f"Changes in FastAPI {version}:"]
    for e in entries:
        lines.append(f"- ({e.category}) {e.description} [PR #{e.pr_number}]({e.pr_url})")
    return "\n".join(lines)


@tool
def check_deprecated(term: str) -> str:
    """Check whether a FastAPI feature/API/parameter has been deprecated or removed.

    Searches the full changelog history for the given term (e.g. "ORJSONResponse",
    "regex parameter") and reports any entries that mention deprecation or removal,
    alongside other mentions for context.
    """
    if not CHANGELOG_PATH.exists():
        return _COLD_START_MESSAGE
    matches = _search(term, parse_changelog())
    if not matches:
        return f"No changelog mentions found for '{term}'. It may never have changed, or the term doesn't match changelog wording."

    deprecation_matches = [
        e for e in matches if "deprecat" in e.description.lower() or "remov" in e.description.lower()
    ]

    lines = [f"Changelog search for '{term}': {len(matches)} total mention(s)."]
    if deprecation_matches:
        lines.append("\nDeprecation/removal-related mentions (most recent first):")
        for e in deprecation_matches:
            date = e.date or "unreleased"
            lines.append(f"- [{e.version}, {date}] {e.description} [PR #{e.pr_number}]({e.pr_url})")
    else:
        lines.append("\nNo deprecation/removal language found in any mention - likely not deprecated.")

    other = [e for e in matches if e not in deprecation_matches][:5]
    if other:
        lines.append("\nOther mentions for context:")
        for e in other:
            date = e.date or "unreleased"
            lines.append(f"- [{e.version}, {date}] {e.description}")

    return "\n".join(lines)


if __name__ == "__main__":
    entries = parse_changelog()
    print(f"Parsed {len(entries)} changelog entries across "
          f"{len({e.version for e in entries})} versions")
    print()
    print("Sample entry:", entries[0])
