from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from src.services.base_service import BaseService
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from src.config import BGE_RERANKER_PATH
import streamlit as st

class RerankingService(BaseService):
    """重排序服务：支持粗排+精排两阶段"""
    
    def __init__(self, coarse_top_k: int = 10, fine_top_k: int = 3):
        super().__init__("重排序服务")
        self.coarse_top_k = coarse_top_k  # 粗排保留数量
        self.fine_top_k = fine_top_k      # 精排保留数量
        self.device = None
        self.tokenizer = None
        self.model = None
    
    def initialize(self) -> bool:
        """初始化重排序服务"""
        try:
            with st.spinner(f"正在初始化{self.service_name}..."):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.tokenizer = AutoTokenizer.from_pretrained(str(BGE_RERANKER_PATH))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(BGE_RERANKER_PATH))
                self.model.to(self.device)
                self.is_initialized = True
                st.success(f"{self.service_name}初始化完成")
                return True
                
        except Exception as e:
            st.error(f"{self.service_name}初始化失败: {str(e)}")
            return False
    
    def process(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """两阶段重排序：粗排 + 精排"""
        if not documents:
            return {
                'coarse_ranked': [],
                'fine_ranked': [],
                'stage_info': {
                    'input_count': 0,
                    'coarse_count': 0,
                    'fine_count': 0
                }
            }
        
        # 阶段1：粗排（快速筛选）
        coarse_ranked = self._coarse_ranking(query, documents)
        
        # 阶段2：精排（精细重排序）
        fine_ranked = self._fine_ranking(query, coarse_ranked)
        
        return {
            'coarse_ranked': coarse_ranked,
            'fine_ranked': fine_ranked,
            'stage_info': {
                'input_count': len(documents),
                'coarse_count': len(coarse_ranked),
                'fine_count': len(fine_ranked)
            }
        }
    
    def _coarse_ranking(self, query: str, documents: List[Document]) -> List[Document]:
        """粗排：基于简单特征的快速排序"""
        scored_docs = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            score = 0
            content_lower = doc.page_content.lower()
            
            # 简单的词匹配得分
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
            
            # 长度偏好（适中长度得分更高）
            content_len = len(doc.page_content)
            if 100 <= content_len <= 1000:
                score += 2
            elif content_len < 100:
                score -= 1
            
            # 标题匹配加分
            title = doc.metadata.get('title', '')
            if title and any(word in title.lower() for word in query_words):
                score += 5
            
            scored_docs.append((doc, score))
        
        # 按分数排序并返回top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.coarse_top_k]]
    
    def _fine_ranking(self, query: str, documents: List[Document]) -> List[Document]:
        """精排：使用深度模型的精确重排序"""
        if not documents or not self.model:
            return documents[:self.fine_top_k]
        
        try:
            # 准备查询-文档对
            pairs = [(query, doc.page_content) for doc in documents]
            
            # 批处理
            batch_size = 4
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # 编码
                features = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                # 计算得分
                with torch.no_grad():
                    scores = self.model(**features).logits.squeeze()
                    
                if len(batch_pairs) == 1:
                    scores = [scores.item()]
                else:
                    scores = scores.cpu().tolist()
                    
                all_scores.extend(scores)
            
            # 组合并排序
            scored_docs = list(zip(documents, all_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in scored_docs[:self.fine_top_k]]
            
        except Exception as e:
            st.error(f"精排序失败: {str(e)}")
            return documents[:self.fine_top_k]