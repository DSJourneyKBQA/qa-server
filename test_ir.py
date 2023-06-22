from intent_recognition import predict

dataset = [
    {
        'question': '什么是分布式',
        'intent': 1,
    },
    {
        'question': '分布式设备是什么',
        'intent': 1,
    },
    {
        'question': 'Vue怎么用',
        'intent': 2,
    },
    {
        'question': 'evu怎么用',
        'intent': 2,
    },
    {
        'question': 'React包括啥',
        'intent': 3,
    },
    {
        'question': 'rect里有什么',
        'intent': 3,
    },
    {
        'question': 'ract里有什么',
        'intent': 3,
    },
    {
        'question': '学完CSS后，应该继续学习什么',
        'intent': 4
    },
    {
        'question': '学Vue前，应该学习什么',
        'intent': 5
    },
    {
        'question': 'Vue的条件渲染是什么',
        'intent': 6
    },
    {
        'question': '条件渲染的v-if怎么用',
        'intent': 7
    },
    {
        'question': 'HTML的标签包括了什么',
        'intent': 8
    },
    {
        'question': '你们一个个都身怀绝技',
        'intent': -1
    },
    {
        'question':'经典模型 DeepFM 原理详解及代码实践',
        'intent': -1
    },
    {
        'question':'大人时代变了',
        'intent': -1
    },
    {
        'question':'你妈什么时候死啊',
        'intent': -1
    }
]
passall = True
results = []
fail = []
for data in dataset:
    res = predict(data['question'].lower())
    if res['prob'] < 0.5:
        res['intent'] = -1
        res['prob'] = -1
    results.append(res)
    if res['intent'] == data['intent']:
        print(f'test [{data["question"]}] pass, prob: {res["prob"]}')
    else:
        print(f'test [{data["question"]}] fail, expect {data["intent"]}, got {res["intent"]}')
        fail.append(data)
        passall = False
        
if passall:
    print(f'all pass, avg prob: {sum([res["prob"] for res in results if res["prob"]!= -1])/len([res for res in results if res["prob"]!= -1])}, max prob: {max([res["prob"] for res in results])}' )
else:
    print('some fail, detail:')
    for data in fail:
        print(f'fail [{data["question"]}]')