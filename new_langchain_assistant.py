import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="You are a helpful assistant",
    tools=[],
    model="gpt-3.5-turbo-1106",
    # model="gpt-4-1106-preview",
)
d = {"content": "Translate below to Korean:\n---\n1. An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\n2. Planetary days will become shorter."}
# output = interpreter_assistant.invoke(d)


async def run_concurrently():
    tasks = [interpreter_assistant.ainvoke(d) for i in range(30)]
    return await tqdm_asyncio.gather(*tasks)


result = asyncio.run(run_concurrently())
print(result)
