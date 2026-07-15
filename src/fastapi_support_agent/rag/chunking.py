"""Load and chunk the FastAPI docs corpus into citeable pieces.

Deliberately excludes release-notes.md - that file is a changelog, not general
documentation. It gets its own structured parser for the deprecation/version
lookup tool (M4) rather than being chunked for semantic search here.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "fastapi-docs" / "docs" / "en" / "docs"
)
DOCS_SITE_BASE = "https://fastapi.tiangolo.com"
EXCLUDE_FILES = {"release-notes.md"}

# MkDocs writes headers like "## Query Parameters { #query-parameters }" - the
# { #anchor-id } part is a slug hint for the site generator, not real content.
HEADER_ATTR_LIST_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def file_path_to_url(md_path: Path) -> str:
    """Map a docs/en/docs/... markdown file path to its live fastapi.tiangolo.com URL."""
    rel = md_path.relative_to(DOCS_ROOT)
    rel = rel.parent if rel.name == "index.md" else rel.with_suffix("")
    url_path = str(rel).replace("\\", "/")
    return f"{DOCS_SITE_BASE}/{url_path}/" if url_path != "." else f"{DOCS_SITE_BASE}/"


def load_and_chunk_docs() -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    all_chunks: list[Document] = []
    for md_path in sorted(DOCS_ROOT.rglob("*.md")):
        # Leading-underscore files are internal tooling (e.g. _llm-test.md is
        # for testing the docs' translation pipeline), not user-facing content.
        if md_path.name in EXCLUDE_FILES or md_path.name.startswith("_"):
            continue

        text = md_path.read_text(encoding="utf-8")
        source_url = file_path_to_url(md_path)
        source_rel = str(md_path.relative_to(DOCS_ROOT))

        for header_doc in header_splitter.split_text(text):
            section = " > ".join(
                HEADER_ATTR_LIST_RE.sub("", v)
                for k, v in header_doc.metadata.items()
                if k in ("h1", "h2", "h3")
            )
            for sub_text in char_splitter.split_text(header_doc.page_content):
                all_chunks.append(
                    Document(
                        page_content=sub_text,
                        metadata={
                            "source_file": source_rel,
                            "section": section or source_rel,
                            "url": source_url,
                        },
                    )
                )

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_docs()
    print(f"Loaded {len(chunks)} chunks from docs corpus")
    print("\nExample chunk:")
    print(chunks[0].metadata)
    print(chunks[0].page_content[:200])
