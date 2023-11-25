import pickle
import time
from pathlib import Path
import itertools
import re

from utils import load_bobj
from batched_chatgpt import call_chatgpt
import prompt_maker

CHATGPT_VERSION_NAME = "gpt-3.5-turbo-1106"
# CHATGPT_VERSION_NAME = "gpt-4-1106-preview"
CHATGPT_SAVE_DIR = "./chatgpt_results/"
CHATGPT_CHUNK_SIZE = 30
TIMEOUT_EACH = 20
SLEEP_BETWEEN_CHUNK = 5
DATASET_NAME = ['arc', 'mmlu', 'truthfulqa'][0]
DEBUG = False
USE_API = True


def main():
    prompts = getattr(prompt_maker, f"make_{DATASET_NAME}_prompt")()
    if DEBUG:
        prompts = prompts[:10]

    file_dir = Path(CHATGPT_SAVE_DIR) / f"{DATASET_NAME}.pkl"

    # chatgpt ranking with paid API, and save pickled files.
    if USE_API:
        print("USE_API True... initializing ChatGPT API...")
        time.sleep(3)  # for emergency stopping the program
        file_dir.parent.mkdir(parents=True, exist_ok=True)
        resp = call_chatgpt(  # this method auto saves the results
            model_name=CHATGPT_VERSION_NAME,
            chunk_size=CHATGPT_CHUNK_SIZE,
            timeout_each=TIMEOUT_EACH,
            sleep_between_chunk=SLEEP_BETWEEN_CHUNK,
            human_message=prompts,
            system_message=['You are expert of Korean language. Translate this conversations to Korean.'] * len(prompts),
            pkl_path=file_dir,
            verbose=True
        )
    else:
        # load pickled file
        resp = load_bobj(file_dir)

    print("\n\n==printing examples...==\n\n")
    for i, s in enumerate(resp[:10]):
        print(prompts[i])
        print(s)
        print()
    print("===========================")


if __name__ == '__main__':
    main()
