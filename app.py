import time,sys,json

from flask import Flask, request
from flask_cors import CORS
from gevent import pywsgi

from utils import Result as R,UnitBot
from intent_recognition import predict,similarity_match
from graph_query import GraphQuery

UNIT_CLIENT_ID = 'Z7KiGkCWNG5WzNUQoKtLDxbh'
UNIT_CLIENT_SECRET = 'rnGquVn3O9FlQONPuUQlmBLGHfX72xcI'
UNIT_BOT_ID = 'S86288'


app = Flask('study')
cors=CORS(app, resources={r"/api/*": {"origins": "*"}})
query = GraphQuery()
unit = UnitBot(UNIT_CLIENT_ID,UNIT_CLIENT_SECRET,UNIT_BOT_ID)


def answer_impl(question:str):
    predict_start = time.time()
    predict_result = predict(question)
    print('predict time: {}s'.format(time.time() - predict_start))
    if predict_result['prob'] < 0.6:
        return R.ok(unit.chat(question),predict_result)
        # return R.ok('非常抱歉，我没能理解你想问什么，我会继续努力学习的',predict_result)
    if predict_result['intent'] == 1: # 知识点描述
        if len(predict_result['entitys']) >0:
            return R.ok(query.get_desc(predict_result['entitys'][0]),predict_result)
    elif predict_result['intent'] == 3: # 知识点的子知识点
        if len(predict_result['entitys']) >0:
            return R.ok(query.get_children(predict_result['entitys'][0]),predict_result)
    elif predict_result['intent'] == 4: # 后置知识点
        if len(predict_result['entitys']) >0:
            return R.ok(query.get_next(predict_result['entitys'][0]),predict_result)
    elif predict_result['intent'] == 5: # 前置知识点
        if len(predict_result['entitys']) >0:
            return R.ok(query.get_require(predict_result['entitys'][0]),predict_result)
    elif predict_result['intent'] == 6: # 子知识点描述
        if len(predict_result['entitys']) >=2:
            return R.ok(query.get_child_desc(predict_result['entitys'][0],predict_result['entitys'][1]),predict_result)
    elif predict_result['intent'] == 8: # 知识点的子知识点的子知识点
        if len(predict_result['entitys']) >=2:
            return R.ok(query.get_children(predict_result['entitys'][0],predict_result['entitys'][1]),predict_result)
    # elif predict_result['intent'] == 10: # 问候
    #     return R.ok('你好呀，我是分布式学习问答机器人，很高兴认识你',predict_result)
    # return R.ok(similarity_match.similarity_match(predict_result['intent'],question),predict_result)
    return R.ok('非常抱歉，我没能理解你想问什么，我会继续努力学习的',predict_result)

@app.route('/api/ping')
def ping():
    return R.ok('pong')

@app.route('/api/answer', methods=['POST'])
def answer():
    req_form = request.form
    question = req_form.get('question','').strip().lower()
    if question == '':
        return R.ok('请输入问题哦')
    return answer_impl(question)
    
    # return R.ok(f'识别到意图{res["intent"]}',res)
    
@app.route('/api/roadmap')
def get_roadmap():
    res = query.get_roadmap()
    return R.ok(data=res)

@app.route('/api/roadmap/<entity>')
def get_entity_roadmap(entity):
    res = query.get_entity_roadmap(entity)
    return R.ok(data=res)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        app.run('0.0.0.0',3001,debug=True)
    else:
        server = pywsgi.WSGIServer(('0.0.0.0', 3001), app)
        server.serve_forever()