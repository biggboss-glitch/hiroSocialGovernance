import requests

def check_openrouter_balance(api_key):
    url = "https://openrouter.ai/api/v1/credits"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Referer": "https://your-site.com", 
        "X-Title": "Balance Check Script"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # 检查请求是否成功
        
        data = response.json().get("data", {})
        
        total_credits = data.get("total_credits", 0) # 总充值/赠送金额
        total_usage = data.get("total_usage", 0)     # 总消耗金额
        
        # 计算剩余余额
        balance = total_credits - total_usage
        
        print(f"💰 总额度 (Total Credits): ${total_credits}")
        print(f"📉 已使用 (Total Usage):   ${total_usage}")
        print(f"💵 剩余余额 (Balance):     ${balance:.4f}")
        
        return balance

    except requests.exceptions.RequestException as e:
        print(f"查询失败: {e}")
        return None

# 替换你的 API Key
MY_API_KEY = ""
check_openrouter_balance(MY_API_KEY)