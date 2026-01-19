#!/usr/bin/env python3
"""
测试 Anthropic API 是否可用
"""
import requests
import json
import os

# API 配置
API_BASE_URL = "https://api.xstx.info"
API_TOKEN = "sk-GIsXmLwrKnRNeVK9uqdJMpoz964g6rx7JkFBSdTiEnjVnNPB"
MODEL = "claude-opus-4-5-20251101"

def test_api():
    """测试API是否可用"""
    url = f"{API_BASE_URL}/v1/messages"

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "x-api-version": "1"
    }

    payload = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello! This is a test message. Please respond with a short greeting."
            }
        ]
    }

    print(f"测试 API: {url}")
    print(f"模型: {MODEL}")
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print("✅ API 调用成功!")
            print(f"响应内容: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ API 调用失败!")
            print(f"响应内容: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
