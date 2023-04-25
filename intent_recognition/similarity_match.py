import json

import hanlp 

sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)

def similarity_match(intent, question):
    std_intent = json.load(open('intent_recognition/data/std_intent.json', 'r', encoding='utf-8'))
    query:str = std_intent.get(f'intent-{intent}')
    if not query:
        return 'intent not support'
    entitys = json.load(open('build_kg/data/entitys.json', 'r', encoding='utf-8'))
    
    for entity in entitys:
        res = sts([query.replace('kp',entity.lower()),question.lower() ])
        print(f'{question}|{query.replace("kp",entity)}:{res}')
        if res > 0.6:
            return query.replace('kp',entity)
    return 'not found'