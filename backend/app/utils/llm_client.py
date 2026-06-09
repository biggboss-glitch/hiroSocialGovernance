"""
LLM客户端封装
统一使用OpenAI格式调用
Supports primary LLM and a fast "boost" LLM for speed-critical paths.
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .logger import get_logger

logger = get_logger('mirofish.llm_client')


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    @classmethod
    def create_boost(cls) -> 'LLMClient':
        """Create a fast LLM client using the boost/Groq config if available.
        Falls back to the primary LLM if boost is not configured."""
        boost_key = getattr(Config, 'LLM_BOOST_API_KEY', None)
        boost_url = getattr(Config, 'LLM_BOOST_BASE_URL', None)
        boost_model = getattr(Config, 'LLM_BOOST_MODEL_NAME', None)
        
        if boost_key and boost_url and boost_model:
            logger.info(f"Using BOOST LLM: {boost_model} @ {boost_url}")
            return cls(api_key=boost_key, base_url=boost_url, model=boost_model)
        else:
            logger.info("No BOOST LLM configured, falling back to primary LLM")
            return cls()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Fallback: strip response_format if it caused the error
            if response_format:
                logger.warning(f"LLM call failed with response_format, retrying without it. Error: {e}")
                kwargs.pop("response_format")
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise Exception(f"{str(e2)} (Base URL: {self.base_url})")
            else:
                raise Exception(f"{str(e)} (Base URL: {self.base_url})")
                
        content = response.choices[0].message.content
        # Strip <think>...</think> blocks (Qwen 3.5, DeepSeek, etc.)
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        return self._parse_json_response(response)
    
    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM output, handling markdown fences,
        think blocks, and other common quirks."""
        cleaned = response.strip()
        
        # Strip any remaining <think> blocks that weren't caught
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned).strip()
        
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Fallback: find the first { ... } block using brace matching
        start = cleaned.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
        
        raise ValueError(f"LLM返回的JSON格式无效: {cleaned[:500]}")
