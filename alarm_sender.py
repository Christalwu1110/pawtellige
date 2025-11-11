# alarm_sender.py
import requests
import json
import os

class AlarmSender:
    def __init__(self):
        print("报警发送器已初始化。")

    def send_alarm(self, message_title, message_body):
        """
        发送报警通知。这是一个抽象方法，具体实现由子类或配置决定。
        你可以替换为真实的报警API调用（微信、钉钉、短信等）。
        :param message_title: 报警标题
        :param message_body: 报警内容
        """
        # 🚨 默认行为：打印到控制台
        print(f"\n--- 🚨🚨🚨 报警！🚨🚨🚨 ---")
        print(f"标题: {message_title}")
        print(f"内容: {message_body}")
        print(f"------------------------")

    def send_wechat_webhook_alarm(self, message_title, message_body, webhook_url):
        """
        通过企业微信群机器人发送报警。
        需要配置企业微信群机器人，获取Webhook URL。
        """
        headers = {'Content-Type': 'application/json'}
        data = {
            "msgtype": "text",
            "text": {
                "content": f"{message_title}\n{message_body}"
            }
        }
        try:
            response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            if result.get('errcode') == 0:
                print("企业微信报警发送成功。")
            else:
                print(f"企业微信报警发送失败: {result.get('errmsg')}")
        except requests.exceptions.RequestException as e:
            print(f"发送企业微信报警时发生网络错误或请求失败: {e}")
        except Exception as e:
            print(f"发送企业微信报警时发生未知错误: {e}")

    def send_dingtalk_webhook_alarm(self, message_title, message_body, webhook_url, secret=None):
        """
        通过钉钉群机器人发送报警。
        需要配置钉钉群机器人，获取Webhook URL，如果设置了安全设置，还需要secret。
        """
        headers = {'Content-Type': 'application/json'}
        
        # 钉钉机器人安全设置：关键词、IP白名单或签名
        # 如果你设置了签名，需要计算sign
        if secret:
            import hmac
            import hashlib
            import base64
            import urllib.parse
            timestamp = str(round(time.time() * 1000))
            string_to_sign = f'{timestamp}\n{secret}'
            hmac_code = hmac.new(string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
            sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
            webhook_url = f"{webhook_url}&timestamp={timestamp}&sign={sign}"

        data = {
            "msgtype": "text",
            "text": {
                "content": f"{message_title}\n{message_body}"
            }
        }
        try:
            response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            if result.get('errcode') == 0:
                print("钉钉报警发送成功。")
            else:
                print(f"钉钉报警发送失败: {result.get('errmsg')}")
        except requests.exceptions.RequestException as e:
            print(f"发送钉钉报警时发生网络错误或请求失败: {e}")
        except Exception as e:
            print(f"发送钉钉报警时发生未知错误: {e}")


# 导入 time 模块，因为它在 dingtalk 报警方法中被用到
import time

if __name__ == "__main__":
    sender = AlarmSender()

    # --- 演示报警发送 ---

    print("\n--- 演示默认控制台报警 ---")
    sender.send_alarm("测试报警", "这是一个测试消息，模拟狗狗异常行为。")

    # --- 演示企业微信报警 (需要真实配置) ---
    # WECHAT_WEBHOOK_URL = os.environ.get("WECHAT_WEBHOOK_URL", "YOUR_WECHAT_WEBHOOK_URL_HERE")
    # if WECHAT_WEBHOOK_URL != "YOUR_WECHAT_WEBHOOK_URL_HERE":
    #     print("\n--- 演示企业微信报警 ---")
    #     sender.send_wechat_webhook_alarm(
    #         "狗狗行为异常提醒",
    #         "狗狗长时间没有活动了！请尽快查看。",
    #         WECHAT_WEBHOOK_URL
    #     )
    # else:
    #     print("\n跳过企业微信报警演示：未配置 WECHAT_WEBHOOK_URL。")

    # --- 演示钉钉报警 (需要真实配置) ---
    # DINGTALK_WEBHOOK_URL = os.environ.get("DINGTALK_WEBHOOK_URL", "YOUR_DINGTALK_WEBHOOK_URL_HERE")
    # DINGTALK_SECRET = os.environ.get("DINGTALK_SECRET", None) # 如果钉钉机器人设置了签名，则需要secret
    # if DINGTALK_WEBHOOK_URL != "YOUR_DINGTALK_WEBHOOK_URL_HERE":
    #     print("\n--- 演示钉钉报警 ---")
    #     sender.send_dingtalk_webhook_alarm(
    #         "狗狗行为异常",
    #         "发现狗狗长时间进食，请检查是否健康。",
    #         DINGTALK_WEBHOOK_URL,
    #         DINGTALK_SECRET
    #     )
    # else:
    #     print("\n跳过钉钉报警演示：未配置 DINGTALK_WEBHOOK_URL。")

    print("\n--- 报警发送器演示完毕 ---")
    