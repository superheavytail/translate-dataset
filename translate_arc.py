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

from utils import pickle_bobj, load_bobj
import prompt_maker

CHATGPT_VERSION_NAME = ...  # to be determined in runtime
CHATGPT_CHUNK_SIZE = 60


@dataclass
class ChatGPTArguments:
    dataset_name: str
    chatgpt_save_dir: str
    chatgpt_version_name: str
    task_desc: str
    use_api: bool = False
    debug: bool = False


async def async_generate(llm, prompt):
    resp = await llm.agenerate([[HumanMessage(content=prompt)]])
    return resp


async def generate_concurrently(prompts):
    # llm = ChatOpenAI(max_tokens=50)
    llm = ChatOpenAI(model_name=CHATGPT_VERSION_NAME, temperature=0.5)
    tasks = [async_generate(llm, prompt) for prompt in prompts]
    return await tqdm_asyncio.gather(*tasks)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_chatgpt_output_async(task, prompts, save_dir=".", time_sleep=30):
    if not os.environ['OPENAI_API_KEY']:
        print("os.environ['OPENAI_API_KEY'] not exists!")
        sys.exit(-1)

    time_str = datetime.datetime.now().strftime('%d%H%M')
    iters = list(chunks(prompts, CHATGPT_CHUNK_SIZE))
    for i, chunk in enumerate(iters):
        print(f"{i}th iterate")
        results = asyncio.run(generate_concurrently(chunk))
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
        get_chatgpt_output_async(task, prompts, save_dir)
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


CONSOLE_DEBUG = False
DATASET_NAME = ['arc', 'mmlu', 'truthfulqa'][0]


def main():
    # for console debugging, set args manually in raw code

    print(f"{CONSOLE_DEBUG}")
    if CONSOLE_DEBUG:
        chatgpt_args = namedtuple("chatgpt_args",
                               ['chatgpt_save_dir', 'chatgpt_version_name', 'task_desc',
                                'use_api', 'debug'])(
            chatgpt_save_dir="./chatgpt_results/",
            chatgpt_version_name="gpt-3.5-turbo-0613",  # TODO change this!
            task_desc="translate",
            debug=False,
            use_api=True
        )
    else:
        parser = HfArgumentParser((ChatGPTArguments,))
        chatgpt_args, *_ = parser.parse_args_into_dataclasses()

    global CHATGPT_VERSION_NAME
    CHATGPT_VERSION_NAME = chatgpt_args.chatgpt_version_name

    prompts = getattr(prompt_maker, f"get_{DATASET_NAME}")

    save_dir = Path(chatgpt_args.chatgpt_save_dir) / DATASET_NAME
    task_desc = chatgpt_args.task_desc

    # Do chatgpt ranking with paid API, and save pickled files.
    if chatgpt_args.use_api:
        print("use_api True... initializing ChatGPT ranking...")
        time.sleep(3)  # for emergency stopping the program
        save_dir.mkdir(parents=True, exist_ok=True)
        assert not any(save_dir.glob("*.pkl"))
        do_chatgpt_async(task_desc, prompts, save_dir)

    # Load pickled files which contain chatgpt-ranked data.
    chatgpt_output = load_chatgpt_result(task_desc, save_dir)
    chatgpt_str = [e.generations[0][0].text for e in chatgpt_output]

    print("\n\n==printing examples...==\n\n")
    for i, s in enumerate(chatgpt_str[:3]):
        print(prompts[i])
        print(s)
        print()
    print("===========================")


if __name__ == '__main__':
    main()
