from pathlib import Path
from typing import List

import re

import pandas as pd
from pydantic import BaseModel, field_validator

from openreward.environments import Environment, tool, JSONObject, ToolOutput, TextBlock
from math_verify import parse, verify


## Math answer parsing (from math environment pattern)

def verify_math_answer(answer: str, reference: str) -> bool:
    """Verify that a submitted answer matches the reference answer.

    math_verify.verify is not symmetric (it may rewrite the gold side, e.g. an
    inequality into an interval), so the reference goes first, as gold.
    """
    return verify(parse_answer(reference), parse_answer(answer))


def parse_answer(answer: str) -> list:
    """Parse math answer with LaTeX handling."""
    parsed = parse(answer)
    if not parsed:
        parsed = parse(f"$ {answer} $")
    return parsed


## Data Models

class TaskSpec(BaseModel):
    id: str
    prompt: str
    solution: str


_INT_RE = re.compile(r"[+-]?\d+\.?")
_DEC_RE = re.compile(r"[+-]?\d*\.\d+")


class AnswerParams(BaseModel):
    """The agent's final answer.

    A JSON number is accepted as well as a string (5 and "5" are the same
    answer). Plain numeric strings are normalised (" +05 " -> "5", "7." -> "7",
    "+2.50" -> "2.50"); any other string (fraction, expression, LaTeX, tuple)
    is only stripped and otherwise left exactly as written.
    """

    answer: str

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_number(cls, v):
        if isinstance(v, bool):
            return v  # not a number answer: let validation reject it
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
        if isinstance(v, str):
            s = v.strip()
            if _INT_RE.fullmatch(s):
                try:
                    return str(int(s.rstrip(".")))
                except ValueError:  # digit string beyond Python's int limit
                    return s
            if _DEC_RE.fullmatch(s):
                return s.lstrip("+")
            return s
        return v


## Dataset Loading

_orwd_path = Path("/orwd_data/train.parquet")
_local_path = Path(__file__).parent / "data" / "train.parquet"
_data_path = _orwd_path if _orwd_path.exists() else _local_path

train_tasks: list[dict] = pd.read_parquet(_data_path).to_dict(orient="records")


## Environment


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class DAPOMath(Environment):
    """DAPO-Math-17k: competition-level math problems with rule-based verification."""

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        # Graded submissions this session. Only the first is rewarded: an
        # uncapped tool would let the agent resubmit after a wrong answer.
        self.submitted = 0

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(
            text=self.config.prompt + "\n\nSubmit your final answer using the answer tool."
        )]

    @tool
    async def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit your final answer for evaluation."""
        if self.submitted > 0:
            return ToolOutput(
                metadata={"already_submitted": True, "submission_count": self.submitted},
                blocks=[TextBlock(text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # An empty answer, or one with no number or expression to parse, is
        # never compared with the solution, so it is not the graded attempt.
        if not parse_answer(params.answer):
            return ToolOutput(
                metadata={"error": "unparseable_answer"},
                blocks=[TextBlock(text="Your answer is empty or could not be parsed as a number or "
                                       "expression, so nothing was graded. Submit your final number "
                                       "or expression.")],
                reward=0,
                finished=False,
            )

        correct = verify_math_answer(params.answer, self.config.solution)
        reward = 1 if correct else 0

        self.submitted += 1

        return ToolOutput(
            metadata={"correct": correct},
            blocks=[TextBlock(text=f"{'Correct' if correct else 'Incorrect'}. Reward: {reward}")],
            reward=reward,
            finished=True,
        )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return train_tasks
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]
