import asyncio
import time
from pprint import pprint
from multiprocessing import Pool, TimeoutError, Queue, Process

from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

CHUNK_SIZE = 30


def process_chunk_element(messages):
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.8, request_timeout=5)
    res = chat.invoke(messages)
    return res


def process_chunk_element2(i, queue, item):
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.8, request_timeout=5)
    res = chat.invoke(item)
    queue.put((i, res))


def process_chunk(chunk, timeout=15):
    with Pool(len(chunk)) as pool:
        result_futures = [pool.apply_async(process_chunk_element, (item,)) for item in chunk]
        results = []
        for future in result_futures:
            try:
                result = future.get(timeout)
                results.append(result)
            except TimeoutError:
                print("timeout!")
                timeout = 0  # tolerate only first timeout
                results.append(None)
                # Here I have to kill the process
    return results


def process_chunk2(chunk, timeout=13):
    processes = []
    output_queue = Queue()
    results = [None] * len(chunk)  # Pre-allocate list for results

    for i, item in enumerate(chunk):
        p = Process(target=process_chunk_element2, args=(i, output_queue, item))
        processes.append(p)
        p.start()

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


def batched_multiprocess_auto_retry(items):
    outputs = [None] * len(items)
    while not all(outputs):
        print("iteration start")
        pprint(outputs)
        remain_inputs = [(i, item) for i, item in enumerate(items) if outputs[i] is None]  # store failed item indices
        remain_indices, remain_items = list(zip(*remain_inputs))
        chunks = [remain_items[i:i + CHUNK_SIZE] for i in range(0, len(remain_items), CHUNK_SIZE)]  # re-chunk remains

        all_results = []
        for chunk in tqdm(chunks):  # tqdm num is the num of chunks
            result = process_chunk2(chunk)
            all_results.extend(result)
        for i in range(len(remain_items)):
            outputs[remain_indices[i]] = all_results[i]
    return outputs


def main():
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content="Translate below sentence into Korean:\n---\n1. An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\n2. Planetary days will become shorter."),
    ]

    messages_list = [messages] * 10

    messages_list = [[
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content=f"say '{i}'"
        )
    ] for i in range(90)]

    responses_list = batched_multiprocess_auto_retry(messages_list)

    print(responses_list)


if __name__ == '__main__':
    main()
