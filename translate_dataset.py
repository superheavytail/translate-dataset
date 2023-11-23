import datetime
import os
import pickle
import asyncio
import time
import sys
from pathlib import Path
import random
import itertools
from typing import List, Union
import re
from dataclasses import dataclass, field
from collections import namedtuple

import torch
from transformers import HfArgumentParser
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm.asyncio import tqdm_asyncio
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

from utils import pickle_bobj, load_bobj
import prompt_maker

CHATGPT_VERSION_NAME = ...  # to be determined in runtime
CHATGPT_CHUNK_SIZE = 10
TIME_SLEEP = 5
CONSOLE_DEBUG = True
DATASET_NAME = ['arc', 'mmlu', 'truthfulqa'][0]
DEBUG = False
USE_API = True


@dataclass
class ChatGPTArguments:
    dataset_name: str
    chatgpt_save_dir: str
    chatgpt_version_name: str
    task_desc: str
    debug: bool = False


async def generate_concurrently(assistant, prompts):
    prompts_dict = [{"content": prompt} for prompt in prompts]
    tasks = [assistant.ainvoke(d) for d in prompts_dict]
    return await tqdm_asyncio.gather(*tasks)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_chatgpt_output_async(task, prompts, save_dir=".", time_sleep=30):
    if not os.environ['OPENAI_API_KEY']:
        print("os.environ['OPENAI_API_KEY'] not exists!")
        sys.exit(-1)

    langchain_assistant = OpenAIAssistantRunnable.create_assistant(
        name="langchain assistant",
        instructions="You are a helpful assistant",
        tools=[],
        model="gpt-3.5-turbo-1106",
        max_retries=10
        # model="gpt-4-1106-preview",
    )

    # save the chatgpt output
    time_str = datetime.datetime.now().strftime('%d%H%M')
    iters = list(chunks(prompts, CHATGPT_CHUNK_SIZE))
    for i, chunk in enumerate(iters):
        print(f"{i}th iterate")
        while True:
            try:
                results = asyncio.run(generate_concurrently(langchain_assistant, chunk))
                break
            except ValueError as e:
                print(e)
                print("retrying...")
        file = Path(save_dir) / f"chatgpt-{task}-{time_str}-{i}.pkl"
        with open(file, 'wb') as f:
            pickle.dump(results, f)
        print(f"{i}th iteration save complete")
        if i + 1 != len(iters):
            time.sleep(time_sleep)  # not sleeps at last iteration


def do_chatgpt_async(task, prompts, save_dir):
    """save chatgpt results to save_dir"""
    save_dir.mkdir(exist_ok=True)
    if save_dir.is_dir():
        get_chatgpt_output_async(task, prompts, save_dir, TIME_SLEEP)
    else:
        raise FileExistsError(f'{save_dir} is not a directory')


def load_chatgpt_result(task, save_dir):
    chatgpt_pickles = list(Path(save_dir).glob(f"chatgpt-{task}-*.pkl"))
    # sort by ascending order
    chatgpt_pickles.sort(key=lambda x:int(re.findall(r'(\d+)\.pkl', str(x))[-1]))
    print("=" * 15)
    print("processing data:")
    print(chatgpt_pickles)
    print("=" * 15)

    datas = []
    for file in chatgpt_pickles:
        with open(file, 'rb') as f:
            datas.append(pickle.load(f))

    return list(itertools.chain(*datas))


def main():
    # for console debugging, set args manually in raw code

    print(f"{CONSOLE_DEBUG}")
    if CONSOLE_DEBUG:
        chatgpt_args = namedtuple("chatgpt_args",
                               ['chatgpt_save_dir', 'chatgpt_version_name', 'task_desc'])(
            chatgpt_save_dir="./chatgpt_results/",
            chatgpt_version_name="gpt-3.5-turbo-1106",  # TODO change this!
            task_desc="translate",
        )
    else:
        parser = HfArgumentParser((ChatGPTArguments,))
        chatgpt_args, *_ = parser.parse_args_into_dataclasses()

    global CHATGPT_VERSION_NAME
    CHATGPT_VERSION_NAME = chatgpt_args.chatgpt_version_name

    prompts = getattr(prompt_maker, f"make_{DATASET_NAME}_prompt")(debug=DEBUG)

    save_dir = Path(chatgpt_args.chatgpt_save_dir) / DATASET_NAME
    task_desc = chatgpt_args.task_desc

    # Do chatgpt ranking with paid API, and save pickled files.
    if USE_API:
        print("USE_API True... initializing ChatGPT API...")
        time.sleep(3)  # for emergency stopping the program
        save_dir.mkdir(parents=True, exist_ok=True)
        assert not any(save_dir.glob("*.pkl"))
        do_chatgpt_async(task_desc, prompts, save_dir)

    # Load pickled files which contain chatgpt-ranked data.
    chatgpt_output = load_chatgpt_result(task_desc, save_dir)
    chatgpt_str = [e[0].content[0].text.value for e in chatgpt_output]

    print("\n\n==printing examples...==\n\n")
    for i, s in enumerate(chatgpt_str[:3]):
        print(prompts[i])
        print(s)
        print()
    print("===========================")


if __name__ == '__main__':
    main()
