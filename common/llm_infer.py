import os
from openai import OpenAI
import cv2
import base64
import time 


# 客户端在调用方初始化，API Key 从环境变量读取（DASHSCOPE_API_KEY / OPENAI_API_KEY）。
# 参考用法：
# client = OpenAI(
#     api_key=os.environ["DASHSCOPE_API_KEY"],
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

def Second_review(client, img, prompt:str):
    """
    二次复审
    """
    try:
        result = {}
        if img is not None:
            # _, buffer = cv2.imencode(".jpg", img)
            b64 = base64.b64encode(img).decode()
            img_data_url = f"data:image/jpeg;base64,{b64}"
            print("prompt:=======================",prompt)
            system_message = """\
            你是一个模型复审专家,评审CNN检测模型的结果,CNN的结果会出现问题,需要你来复审.
            """
            vl_messages=[
            {
            'role': 'system',
            'content': system_message
            },
            {
                "role": "user",
                "content": 
                            [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_data_url}
                                },
                                {
                                    "type": "text", "text": prompt
                                },
                            ]
            }
            ]
            completion = client.chat.completions.create(
                model="qwen3.8-flash",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=vl_messages,
                stream=False,
            )
            result = completion.choices[0].message.content
            return True,result
        else:
            print(f"当前图片帧已丢失")
            return False,result
    except Exception as e:
        raise e

def realtime_llm_infer(client, img_data_url, prompt):
    """
    实时流大模型推理
    ：提示词拼接在此处
    ：

    """
    system_message = """\
    你是一个巡检机器人,从用户上传的图片中识别待检测目标,从用户的提示词中找出待检测目标,请注意图片中可能不存在待检测目标,请直接回复空JSON对象{},以JSON格式输出,包含类别和坐标.
    """
    start_time = time.time()

    try:
        result = {}
        if img_data_url is not None:
            print("prompt:=======================",prompt)

            vl_messages=[
            {
            'role': 'system',
            'content': system_message
            },
            {
                "role": "user",
                "content":
                            [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_data_url}
                                },
                                {
                                    "type": "text", "text": prompt + "仅检测当前用户提示词中目标,如果没有提示词中目标,请直接回复空JSON对象{}"
                                },
                            ]
            }
            ]
            completion = client.chat.completions.create(
                model="qwen3.8-flash",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=vl_messages,
                stream=False,
            )
            end_time = time.time()
            spend_time = end_time - start_time
            result = completion.choices[0].message.content
            print(f"[LLM-TIME] stage=llm_request(网关往返) cost={spend_time:.2f}s")
            return True, result, spend_time
        else:
            print(f"当前图片帧已丢失")
            return False,result
    except Exception as e:
        spend_time = time.time() - start_time
        print(f"[LLM] realtime 请求异常: {e}, spend_time={spend_time:.2f}s")
        return False, None, spend_time
