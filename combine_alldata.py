import json
import random
divorce_data_sentence = 'data/divorce/input.json'
labor_data_sentence = 'data/labor/input.json'
loan_data_sentence = 'data/loan/input.json'
all_data_sentence = 'data/all_data_shuffle/input.json'
divorce_data_tags = 'data/divorce/tags.txt'
labor_data_tags = 'data/labor/tags.txt'
loan_data_tags = 'data/loan/tags.txt'
all_data_tags = 'data/all_data_shuffle/tags.txt'
with open(divorce_data_sentence,'r',encoding='utf-8') as dv,open(all_data_sentence,'w',encoding='utf-8') as all_sentence:
    lines = []
    # DV 写一次
    for line in dv:
        line = json.loads(line)
        lines.append(line)

    # LB写两次
    with open(labor_data_sentence,'r',encoding='utf-8') as lb:
        for line in lb:
            line = json.loads(line)
            lines.append(line)
    with open(labor_data_sentence,'r',encoding='utf-8') as lb:
        for line in lb:
            line = json.loads(line)
            lines.append(line)

    # LN写两次
    with open(loan_data_sentence,'r',encoding='utf-8') as ln:
        for line in ln:
            line = json.loads(line)
            lines.append(line)
    with open(loan_data_sentence, 'r', encoding='utf-8') as ln:
        for line in ln:
            line = json.loads(line)
            lines.append(line)

    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    for line in lines:
        json.dump(line,all_sentence,ensure_ascii=False)
        all_sentence.write('\n')

with open(divorce_data_tags,'r',encoding='utf-8') as dv,open(all_data_tags,'w',encoding='utf-8') as all_tags:
    for line in dv:
        all_tags.write(line)
    with open(labor_data_tags,'r',encoding='utf-8') as lb:
        for line in lb:
            all_tags.write(line)
    with open(loan_data_tags,'r',encoding='utf-8') as ln:
        for line in ln:
            all_tags.write(line)