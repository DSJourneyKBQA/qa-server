import time

from flask import Flask, request
from flask_cors import CORS

from utils import Result as R
from intent_recognition import predict
from graph_query import GraphQuery

app = Flask('study')
cors=CORS(app, resources={r"/api/*": {"origins": "*"}})
query = GraphQuery()

@app.route('/api/ping')
def ping():
    return R.ok('pong')

@app.route('/api/answer', methods=['POST'])
def answer():
    start = time.time()
    req_form = request.form
    question = req_form.get('question','').strip().lower()
    if question == '':
        return R.ok('请输入问题哦')
    res = predict(question)
    print('predict time: {}s'.format(time.time() - start))
    if res['prob'] < 0.8:
        return R.ok('非常抱歉，我没能理解你想问什么，我会继续努力学习的',res)
    if res['intent'] == 1: # 知识点描述
        if len(res['entitys']) >0:
            return R.ok(query.get_desc(res['entitys'][0]),res)
    elif res['intent'] == 3: # 知识点的子知识点
        if len(res['entitys']) >0:
            return R.ok(query.get_children(res['entitys'][0]),res)
    elif res['intent'] == 4: # 后置知识点
        if len(res['entitys']) >0:
            return R.ok(query.get_next(res['entitys'][0]),res)
    elif res['intent'] == 5: # 前置知识点
        if len(res['entitys']) >0:
            return R.ok(query.get_require(res['entitys'][0]),res)
    elif res['intent'] == 6: # 子知识点描述
        if len(res['entitys']) >=2:
            return R.ok(query.get_child_desc(res['entitys'][0],res['entitys'][1]),res)
    elif res['intent'] == 8: # 知识点的子知识点的子知识点
        if len(res['entitys']) >=2:
            return R.ok(query.get_children(res['entitys'][0],res['entitys'][1]),res)
    elif res['intent'] == 10: # 问候
        return R.ok('你好呀，我是分布式学习问答机器人，很高兴认识你',res)
    return R.ok('非常抱歉，我没能理解你想问什么，我会继续努力学习的',res)
    # return R.ok(f'识别到意图{res["intent"]}',res)

if __name__ == '__main__':
    app.run('0.0.0.0',3001,debug=True)