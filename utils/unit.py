import requests
import time,random

client_id = "Z7KiGkCWNG5WzNUQoKtLDxbh"
client_secret = "rnGquVn3O9FlQONPuUQlmBLGHfX72xcI"

class UnitBot(object):
    def __init__(self,client_id,client_secret,bot_id):
        self.access_token = None
        self.client_id = client_id
        self.client_secret = client_secret
        self.bot_id = bot_id

    def get_access_token(self):
        if self.access_token is not None:
            return self.access_token
        access_token_url = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}'
        response = requests.get(access_token_url)
        if not response.ok:
            return None
        access_token = response.json()['access_token']
        self.access_token = access_token
        return access_token

    def chat(self,question,uid='1'):
        access_token = self.get_access_token()
        if access_token is None:
            return '机器人初始化失败'
        chat_url = f'https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token={access_token}'
        headers = {'content-type': 'application/json'}
        payload = {
            'version': '3.0',
            'service_id': self.bot_id,
            'log_id': f'unit_{int(time.time())}',
            'session_id': '',
            'request':{
                'query': question,
                'terminal_id': '000'
            }
        }
        res = requests.post(chat_url,json=payload,headers=headers)
        if not res.ok:
            print('机器人请求失败')
            return '机器人请求失败'
        json_data = res.json()
        print(json_data)
        if json_data['error_code'] != 0:
            print('机器人请求失败')
            return '机器人请求失败'
        responses = json_data['result']['responses']
        actions = responses[0]['actions']
        reply = random.choice(actions)['say']
        return reply