import json
import sys
import random
import requests
import os
import numpy as np
import torch

def send_msg(msg):
    url = 'https://hooks.slack.com/services/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'
    message = ("Train 완료!!!\n" + msg) 
    title = (f"New Incoming Message :zap:") # 타이틀 입력
    slack_data = {
        "username": "NotificationBot", # 보내는 사람 이름
        "icon_emoji": ":satellite:",
        #"channel" : "#somerandomcahnnel",
        "attachments": [
            {
                "color": "#9733EE",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True