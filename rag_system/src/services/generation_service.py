from typing import List, Dict, Any
from langchain.schema import Document
from src.services.base_service import BaseService
from openai import OpenAI
from src.config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL, SYSTEM_PROMPT
import streamlit as st

class GenerationService(BaseService):
    """生成服务"""
    
    def __init__(self):
        super().__init__("生成服务")
        self.client = None
    
    def initialize(self) -> bool:
        """初始化生成服务"""
        try:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE
            )
            self.is_initialized = True
            st.success(f"{self.service_name}初始化完成")
            return True
            
        except Exception as e:
            st.error(f"{self.service_name}初始化失败: {str(e)}")
            return False
    
    def process(self, query: str, documents: List[Document], 
                conversation_history: List[Dict] = None) -> str:
        """生成回答"""
        if not self.client:
            raise Exception("OpenAI客户端未初始化")
        
        # 准备上下文
        context = self._prepare_context(documents)
        
        # 构建提示词
        if conversation_history:
            # 多轮对话模式
            return self._generate_chat_response(query, context, conversation_history)
        else:
            # 单轮问答模式
            return self._generate_single_response(query, context)
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """准备上下文信息"""
        if not documents:
            return "没有找到相关文档。"
        
        contexts = []
        for idx, doc in enumerate(documents, 1):
            source = doc.metadata.get('filename') or doc.metadata.get('title', '未知来源')
            granularity = doc.metadata.get('granularity', 'paragraph')
            
            context = (
                f"[参考资料{idx}] ({granularity}级别)\n"
                f"来源: {source}\n"
                f"内容: {doc.page_content.strip()}\n"
            )
            contexts.append(context)
        
        return "\n".join(contexts)
    
    def _generate_single_response(self, query: str, context: str) -> str:
        """生成单轮回答"""
        prompt = f"""
参考资料：
{context}

用户问题：{query}

请基于参考资料回答问题，要求：
1. 严格基于参考资料内容
2. 回答结构清晰，有条理
3. 适当引用原文关键信息
4. 如果信息不足，请明确指出
"""
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def _generate_chat_response(self, query: str, context: str, 
                               conversation_history: List[Dict]) -> str:
        """生成多轮对话回答"""
        # 构建系统消息，包含上下文
        system_content = f"{SYSTEM_PROMPT}\n\n当前相关资料：\n{context}"
        
        # 构建完整对话历史
        messages = [{"role": "system", "content": system_content}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1500
        )
        
        return response.choices[0].message.content