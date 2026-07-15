"""Fetch the FastAPI docs corpus (including the changelog) via a shallow, sparse git clone.

Pulls only docs/en/docs/ from fastapi/fastapi@master into data/raw/fastapi-docs/.
Safe to re-run: clones if missing, pulls latest if already present.
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
SPARSE_PATH = "docs/en/docs"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "fastapi-docs"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def fetch_docs() -> None:
    if DEST.exists():
        print(f"{DEST} already exists, pulling latest...")
        run("git", "pull", cwd=DEST)
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} (sparse: {SPARSE_PATH}) into {DEST}...")
    run(
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        REPO_URL,
        str(DEST),
    )
    run("git", "sparse-checkout", "set", SPARSE_PATH, cwd=DEST)
    print("Done.")


if __name__ == "__main__":
    fetch_docs()
