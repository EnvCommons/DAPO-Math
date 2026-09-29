from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel

from openreward.environments import Environment, tool, JSONObject, ToolOutput, TextBlock
from math_verify import parse, verify


## Math answer parsing (from math environment pattern)

def verify_math_answer(answer_one: str, answer_two: str) -> bool:
    """Verify if two math answers are equivalent."""
    return verify(parse_answer(answer_one), parse_answer(answer_two))


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


class AnswerParams(BaseModel):
    answer: str


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
