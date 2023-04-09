import pandas as pd

dataset_name = '0'
label_list = []
text_list = []


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
        label_list.append(dataset_name)
        text_list.append(' '.join(line.replace('\n', '')).replace('k p', 'kp'))

        line = f.readline()

print(label_list)
print(text_list)


data = pd.DataFrame({'label': label_list, 'text': text_list})

data.to_csv('./data/cls_data.csv',index=False)