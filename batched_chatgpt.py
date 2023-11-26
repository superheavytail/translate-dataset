import os
import time
from typing import List, Union
from multiprocessing import Queue, Process
from pathlib import Path
from pprint import pprint

from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from utils import pickle_bobj, get_saving_filename_safely


def process_chunk_element(i, queue, item):
    # chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, request_timeout=5)
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    res = chat.invoke(item)
    queue.put((i, res))


def process_chunk(chunk, timeout=13):
    processes = []
    output_queue = Queue()
    results = [None] * len(chunk)  # Pre-allocate list for results

    for i, item in enumerate(chunk):
        p = Process(target=process_chunk_element, args=(i, output_queue, item))
        processes.append(p)
        p.start()
        time.sleep(0.2)  # restrict too dense api calling

    start_time = time.time()
    completed = 0

    while completed < len(processes) and time.time() - start_time < timeout:
        if not output_queue.empty():
            index, result = output_queue.get()
            results[index] = result
            completed += 1

    # Terminate any remaining processes
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

    return results


def batched_multiprocess_auto_retry(items, chunk_size, timeout_each, sleep_between_chunk, pkl_path, verbose=False):
    """returns list of chatgpt output string

    timeout-ed output be None"""
    pkl_path = get_saving_filename_safely(pkl_path) if pkl_path else None  # if pkl_path, result saved.

    outputs = [None] * len(items)
    while not all(outputs):
        # printing remained queries if the number of remained queries is small
        num_of_remains = outputs.count(None)
        print(f"num of remains: {num_of_remains}") if verbose else ...
        if verbose and num_of_remains <= chunk_size:
            pprint(f"printing remains...:\n{[items[i][1].content for i, o in enumerate(outputs) if o is None]}")

        remain_inputs = [(i, item) for i, item in enumerate(items) if outputs[i] is None]  # store failed item indices
        remain_indices, remain_items = list(zip(*remain_inputs))
        chunks = [remain_items[i:i + chunk_size] for i in range(0, len(remain_items), chunk_size)]  # re-chunk remains

        all_results = []
        for chunk in tqdm(chunks):  # tqdm num is the num of chunks
            result = process_chunk(chunk, timeout_each)
            result = map(lambda x: x.content if x else None, result)
            all_results.extend(result)
        for i in range(len(remain_items)):
            outputs[remain_indices[i]] = all_results[i]

        # save the outputs which may be incomplete
        pickle_bobj(outputs, pkl_path) if pkl_path else None

        time.sleep(sleep_between_chunk)
    return outputs


def call_chatgpt(
        model_name: str,
        chunk_size: int,
        timeout_each: int,
        sleep_between_chunk: int,
        human_message: List[str],
        system_message: Union[str, List[str]] = "You're a helpful assistant",
        pkl_path: Union[Path, str] = None,
        verbose: bool = False) -> List[str]:
    """call batched chatgpt api, and automatically save the responses.

    if pkl_path is not None, then this function automatically save the results to pkl_path."""
    assert isinstance(system_message, list) or isinstance(system_message, str)
    assert isinstance(human_message, list)
    if isinstance(system_message, str):
        system_message = [system_message] * len(human_message)
    assert len(system_message) == len(human_message)
    assert model_name in ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']
    assert os.environ['OPENAI_API_KEY']

    messages_list = [[
        SystemMessage(content=system_message[i]),
        HumanMessage(
            content=human_message[i]
        )
    ] for i in range(len(system_message))]

    resp = batched_multiprocess_auto_retry(
        messages_list, chunk_size, timeout_each, sleep_between_chunk, pkl_path, verbose)
    # resp = [r.content for r in resp]
    return resp


def main():
    raise AssertionError
    sys_msg = "You're a helpful assistant"
    hu_msg = [f"whatis 61441*23?" for i in range(10)]

    resp = call_chatgpt(
        model_name="gpt-3.5-turbo-1106",
        chunk_size=3,
        timeout_each=10,
        system_message=[sys_msg] * 10,
        human_message=hu_msg,
    )

    print(resp)


if __name__ == '__main__':
    main()
