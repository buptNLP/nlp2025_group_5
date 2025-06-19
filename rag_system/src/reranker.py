from typing import List, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from langchain.schema import Document
from src.config import BGE_RERANKER_PATH
import streamlit as st

class Reranker:
    def __init__(self, rerank_top_k: int = 3):
        self.rerank_top_k = rerank_top_k
        self.device = None
        self.tokenizer = None
        self.model = None
        
    def _load_model(self):
        """懒加载模型，仅在需要时初始化"""
        if self.model is None:
            try:
                with st.spinner("加载重排序模型中..."):
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.tokenizer = AutoTokenizer.from_pretrained(str(BGE_RERANKER_PATH))
                    self.model = AutoModelForSequenceClassification.from_pretrained(str(BGE_RERANKER_PATH))
                    self.model.to(self.device)
                    st.success(f"重排序模型加载成功")
            except Exception as e:
                st.error(f"加载重排序模型失败: {str(e)}")
                import traceback
                st.error(f"详细错误信息: {traceback.format_exc()}")
                return False
        return True
        
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """重新排序文档"""
        if not documents:
            return []
            
        # 加载模型（如果尚未加载）
        if not self._load_model():
            return documents[:min(self.rerank_top_k, len(documents))]
        
        try:
            # 准备查询-文档对
            pairs = [(query, doc.page_content) for doc in documents]
            
            # 分批处理以降低内存使用
            batch_size = 8
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # 对输入进行编码
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
                    
                # 如果只有一个样本，确保结果是一维的
                if len(batch_pairs) == 1:
                    scores = [scores.item()]
                else:
                    scores = scores.cpu().tolist()
                    
                all_scores.extend(scores)
            
            # 组合文档和得分
            scored_docs = list(zip(documents, all_scores))
            
            # 按得分降序排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前k个文档
            top_docs = [doc for doc, _ in scored_docs[:self.rerank_top_k]]
            
            return top_docs
            
        except Exception as e:
            st.error(f"重排序过程出错: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 出错时返回原始文档的前k个
            return documents[:min(self.rerank_top_k, len(documents))]