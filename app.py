import time

from flask import Flask, request
from flask_cors import CORS

from utils import Result as R
from intent_recognition import predict

app = Flask('study')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/ping')
def ping():
    return R.ok('pong')


@app.route('/api/answer', methods=['POST'])
def answer():
    start = time.time()
    req_form = request.form
    question = req_form.get('question', '').strip()
    if question == '':
        return R.ok('请输入问题哦')
    res = predict(question)
    print('predict time: {}s'.format(time.time() - start))
    if res['prob'] < 0.9:
        return R.ok('非常抱歉，我没能理解你想问什么，我会继续努力学习的', res)

    if res['intent'] == 10:
        return R.ok('你好呀，我是分布式学习问答机器人，很高兴认识你', res)
    return R.ok(f'识别到意图{res["intent"]}', res)


if __name__ == '__main__':
    print("1234")
    print("1234")
    print("1234")
    print("2342")
    app.run('0.0.0.0', 3001, debug=True)
