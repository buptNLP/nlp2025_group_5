from typing import List, Dict, Any
from langchain.schema import Document
from src.services.base_service import BaseService
from src.hybrid_retriever import HybridRetriever
import streamlit as st

class RetrievalService(BaseService):
    """检索服务"""
    
    def __init__(self, documents: List[Document] = None, top_k: int = 20):
        super().__init__("检索服务")
        self.documents = documents or []
        self.top_k = top_k
        self.retriever = None
    
    def initialize(self) -> bool:
        """初始化检索服务"""
        try:
            if not self.documents:
                st.warning("检索服务：没有文档可用于检索")
                return False
            
            with st.spinner(f"正在初始化{self.service_name}..."):
                self.retriever = HybridRetriever(self.documents, self.top_k)
                self.is_initialized = True
                st.success(f"{self.service_name}初始化完成")
                return True
                
        except Exception as e:
            st.error(f"{self.service_name}初始化失败: {str(e)}")
            return False
    
    def process(self, query: str, alpha: float = 0.7) -> List[Document]:
        """执行检索"""
        if not self.retriever:
            raise Exception("检索器未初始化")
        
        return self.retriever.retrieve(query, alpha)
    
    def update_documents(self, documents: List[Document]):
        """更新文档并重新初始化"""
        self.documents = documents
        self.is_initialized = False
        return self.initialize()