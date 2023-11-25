from translate_dataset import load_chatgpt_result
from prompt_maker import make_arc_prompt

l = load_chatgpt_result("translate-221646", "chatgpt_results/arc")

# print(l[1122])
# print(make_arc_prompt()[1122])
data = {'instruction': [], 'response': []}
error_count = 0
for i, item in enumerate(l):
    txt = item[0].content[0].text.value

    if "\n2. " in txt:
        try:
            inst, resp = txt.split("\n2. ")
        except ValueError:
            continue

        if inst.startswith('1.'):
            inst = inst[2:]
        resp = resp.strip()

        data['instruction'].append(inst)
        data['response'].append(resp)
    else:
        continue

print(len(data['instruction']))
print("===")
for i in range(len(data['instruction'])):
    print("===")
    print(data['instruction'][i])
    print(data['response'][i])
