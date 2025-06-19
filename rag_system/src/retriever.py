from typing import List, Dict, Optional
import faiss
import numpy as np
from langchain.schema import Document
from src.embeddings import CustomEmbeddings
import streamlit as st

class Retriever:
    def __init__(self, documents: List[Document], top_k: int = 10):
        self.documents = documents
        self.top_k = top_k
        self.embeddings = CustomEmbeddings()
        self.index = None
        
        # Only try to create index if we have documents
        if documents:
            try:
                self.index = self._create_index()
                st.success(f"成功创建检索索引，包含 {len(documents)} 个文本块")
            except Exception as e:
                st.error(f"创建检索索引时出错: {str(e)}")
                import traceback
                st.error(f"详细错误信息: {traceback.format_exc()}")
        
    def _create_index(self) -> Optional[faiss.IndexFlatIP]:
        """创建FAISS索引"""
        if not self.documents:
            return None
            
        texts = [doc.page_content for doc in self.documents]
        
        # 进度显示
        with st.spinner("正在生成文本向量..."):
            embeddings = self.embeddings.embed_documents(texts)
        
        if not embeddings:
            st.error("生成文本向量失败")
            return None
            
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        
        # 转换为numpy数组并添加到索引
        with st.spinner("正在构建向量索引..."):
            index.add(np.array(embeddings, dtype=np.float32))
            
        return index
    
    def retrieve(self, query: str) -> List[Document]:
        """检索相关文档"""
        if not self.index or not self.documents:
            st.warning("检索系统未初始化或没有文档可供检索")
            return []
            
        try:
            # 生成查询向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 调整top_k以确保不超过文档数量
            actual_top_k = min(self.top_k, len(self.documents))
            
            # 执行检索
            scores, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), 
                actual_top_k
            )
            
            # 过滤无效索引（以防万一）
            valid_indices = [i for i in indices[0] if 0 <= i < len(self.documents)]
            
            # 返回检索到的文档
            return [self.documents[i] for i in valid_indices]
            
        except Exception as e:
            st.error(f"检索文档时出错: {str(e)}")
            return []