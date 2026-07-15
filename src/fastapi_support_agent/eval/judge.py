"""LLM-as-judge: scores an actual agent answer against a golden-set expected answer."""

from pydantic import BaseModel, Field

from fastapi_support_agent.gateway.client import get_gateway_llm

JUDGE_PROMPT = """You are grading a support agent's answer against a known-correct reference.

Question: {question}

Reference (ground-truth) answer: {expected_answer}

Sources the answer should ideally reference: {expected_sources}

The agent's actual answer:
{actual_answer}

Score the actual answer on two dimensions, 1-5 each:
- correctness: does it convey the same key facts as the reference answer? 5 = fully correct, 1 = wrong/contradicts the reference.
- citation_accuracy: does it cite the expected sources (or clearly equivalent ones)? 5 = cites them clearly, 1 = no relevant citation at all.

Give a brief reason for your scores."""


class JudgeScore(BaseModel):
    correctness: int = Field(description="1-5: factual match with the reference answer")
    citation_accuracy: int = Field(description="1-5: whether expected sources are cited")
    reasoning: str = Field(description="Brief explanation of the scores")


def judge_answer(
    question: str, expected_answer: str, expected_sources: list[str], actual_answer: str
) -> JudgeScore:
    judge_llm = get_gateway_llm().with_structured_output(JudgeScore)
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        expected_sources=", ".join(expected_sources) or "(none specific)",
        actual_answer=actual_answer,
    )
    return judge_llm.invoke(prompt)
