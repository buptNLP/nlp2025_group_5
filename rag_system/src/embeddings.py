from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.config import BGE_MODEL_PATH
import torch
import streamlit as st

class CustomEmbeddings:
    def __init__(self):
        self.model = None
        self.device = None
        self._initialize_model()
        
    def _initialize_model(self) -> bool:
        """初始化嵌入模型"""
        if self.model is not None:
            return True
            
        try:
            st.info(f"正在加载嵌入模型 {BGE_MODEL_PATH}...")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(str(BGE_MODEL_PATH))
            self.model.to(self.device)
            st.success(f"嵌入模型加载成功")
            return True
        except Exception as e:
            st.error(f"加载嵌入模型失败: {e}")
            st.error(f"模型路径: {BGE_MODEL_PATH}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为向量"""
        if not texts:
            return []
            
        if not self._initialize_model():
            st.error("嵌入模型未初始化，无法生成文档向量")
            return []
            
        try:
            # 批处理以减少内存使用
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                with st.spinner(f"正在生成文档向量 ({i+1}-{min(i+batch_size, len(texts))}/{len(texts)})..."):
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            st.error(f"生成文档向量失败: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return []

    def embed_query(self, text: str) -> Optional[List[float]]:
        """将查询转换为向量"""
        if not text.strip():
            return None
            
        if not self._initialize_model():
            st.error("嵌入模型未初始化，无法生成查询向量")
            return None
            
        try:
            with st.spinner("正在生成查询向量..."):
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
                
        except Exception as e:
            st.error(f"生成查询向量失败: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return None