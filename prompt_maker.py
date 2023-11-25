from typing import List
from string import ascii_uppercase

from datasets import load_dataset, concatenate_datasets


def make_arc_prompt() -> List[str]:
    # make from train and validation split
    ds = load_dataset("ai2_arc", 'ARC-Challenge')

    # use only test split (1172 examples)
    # ds = ds['test']

    # use train + valid split
    print("use train + validation + test split...")
    ds = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

    queries = []
    for i, example in enumerate(ds):
        answerkey = example['answerKey']
        label = example['choices']['label']
        index = label.index(answerkey)

        query = example['question'].strip()
        if not (query.endswith('?') or query.endswith('.')):
            query = query + '...'

        queries.append(
            f'A: "{query}"\nB: "{example["choices"]["text"][index]}"')
    return queries


def make_mmlu_prompt():
    # not for use...
    # I will use handcrafted dataset
    mmlu_subsets = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics',
                    'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology',
                    'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology',
                    'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition',
                    'global_facts', 'machine_learning', 'security_studies', 'public_relations',
                    'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine',
                    'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies',
                    'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics',
                    'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law',
                    'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous',
                    'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics',
                    'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics',
                    'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy',
                    'college_biology']
    subsets_for_use = ['astronomy', 'moral_scenarios', 'high_school_mathematics']
    mmlu_d = {}
    for subset_name in mmlu_subsets:
        tmp = load_dataset("lukaemon/mmlu", subset_name)
        mmlu_d[subset_name] = tmp


def make_truthfulqa_prompt() -> List[str]:
    # tru_multiplechoice = load_dataset("truthful_qa", 'multiple_choice')

    # use truthful_qa generation
    tru_generation = load_dataset("truthful_qa", 'generation')['validation']

    queries = []
    for i, example in enumerate(tru_generation):
        queries.append(
            f'A: "{example["question"]}"\nB: "{example["best_answer"]}"'
        )
    return queries
