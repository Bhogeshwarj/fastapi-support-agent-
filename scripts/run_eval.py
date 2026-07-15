"""Run the golden Q&A set through the full agent graph and score with the judge.

Saves a timestamped, git-commit-tagged result file to eval_runs/ (committed to
git, unlike data/) so eval scores can be compared across commits over time.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.types import Command

from fastapi_support_agent.agents.graph import build_agent_graph
from fastapi_support_agent.eval.judge import judge_answer
from fastapi_support_agent.gateway.content import extract_text

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "src" / "fastapi_support_agent" / "eval" / "golden_set.json"
EVAL_RUNS_DIR = Path(__file__).resolve().parent.parent / "eval_runs"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def run_question(app, thread_id: str, question: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", question)]}, config)
    # Auto-approve any HITL interrupt - this is unattended batch eval, not
    # manual review. A real user-facing run would surface this to a human.
    if "__interrupt__" in result:
        result = app.invoke(Command(resume="approve"), config)
    return extract_text(result["messages"][-1].content)


def run_eval() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    app = build_agent_graph()

    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = get_git_commit()
    out_path = EVAL_RUNS_DIR / f"eval_{timestamp}_{commit}.json"

    results = []
    for i, case in enumerate(golden_set):
        print(f"[{i + 1}/{len(golden_set)}] {case['id']}: {case['question']}", flush=True)
        t0 = time.time()
        actual_answer = run_question(app, thread_id=f"eval-{case['id']}", question=case["question"])
        print(f"    (answered in {time.time() - t0:.1f}s, scoring...)", flush=True)
        score = judge_answer(
            question=case["question"],
            expected_answer=case["expected_answer"],
            expected_sources=case["expected_sources"],
            actual_answer=actual_answer,
        )
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "actual_answer": actual_answer,
                "correctness": score.correctness,
                "citation_accuracy": score.citation_accuracy,
                "reasoning": score.reasoning,
                "known_limitation": case.get("known_limitation", False),
            }
        )
        flag = " (known limitation)" if case.get("known_limitation") else ""
        print(f"    correctness={score.correctness}/5 citation={score.citation_accuracy}/5{flag}", flush=True)

        # Save after every question, not just at the end - a long run that gets
        # interrupted shouldn't lose everything computed so far.
        out_path.write_text(json.dumps({"timestamp": timestamp, "commit": commit, "results": results}, indent=2))

    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_citation = sum(r["citation_accuracy"] for r in results) / len(results)
    known_limitation_count = sum(1 for r in results if r["known_limitation"])
    print()
    print(f"Average correctness: {avg_correctness:.2f}/5")
    print(f"Average citation accuracy: {avg_citation:.2f}/5")
    print(f"Known-limitation cases: {known_limitation_count}/{len(results)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    run_eval()
