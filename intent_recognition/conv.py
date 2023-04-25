import random
import pandas as pd

dataset_name = '0'

dataset_list = []


with open('./data/question.txt', 'r',encoding='utf-8') as f:
    line = f.readline()
    while line:
        line = line.replace('\n','')
        if line == '':
            line = f.readline()
            continue
        if line.startswith('dataset'):
            dataset_name = line.split(':')[1]
            line = f.readline()
            continue
        dataset_list.append({
            'label': dataset_name,
            'text': ' '.join(line.replace('\n', '')).replace('k p', 'kp')
        })

        line = f.readline()

dataset_list.sort(key=lambda x: random.random())
label_list = [i['label'] for i in dataset_list]
text_list = [i['text'] for i in dataset_list]
print(label_list)
print(text_list)


data = pd.DataFrame({'label': label_list, 'text': text_list})

data.to_csv('./data/cls_data.csv',index=False)