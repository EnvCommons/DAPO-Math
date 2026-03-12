import asyncio

from tqdm.auto import tqdm
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward()

    environment = or_client.environments.get(
        name="dapo_math",
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split="train")
    print(f"Found {len(tasks)} tasks. Running golden tests...")

    for task in tqdm(tasks):
        async with environment.session(task=task, secrets={}) as session:
            solution = task.task_spec["solution"]
            result = await session.call_tool("answer", {"answer": solution})

            assert result.finished, f"Task not finished"
            assert result.reward == 1.0, f"Expected reward 1.0, got {result.reward} for task {task.task_spec['id']}"

    print("All golden tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
