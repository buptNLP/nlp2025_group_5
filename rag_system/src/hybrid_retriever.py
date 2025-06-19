from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import jieba
from src.embeddings import CustomEmbeddings
import streamlit as st

class HybridRetriever:
    def __init__(self, documents: List[Document], top_k: int = 20):
        """
        混合检索器：结合向量检索和BM25
        支持段落级和文档级检索
        """
        self.documents = documents
        self.top_k = top_k
        self.embeddings = CustomEmbeddings()
        
        # 分离段落级和文档级内容
        self.paragraph_docs, self.document_docs = self._separate_granularity()
        
        # 初始化检索组件
        self.vector_index = None
        self.bm25_model = None
        self.doc_vector_index = None
        self.doc_bm25_model = None
        
        if documents:
            self._initialize_retrievers()
    
    def _separate_granularity(self) -> Tuple[List[Document], List[Document]]:
        """分离段落级和文档级内容"""
        paragraph_docs = []
        document_groups = {}
        
        # 按来源文档分组
        for doc in self.documents:
            source = doc.metadata.get('filename') or doc.metadata.get('title', 'unknown')
            if source not in document_groups:
                document_groups[source] = []
            document_groups[source].append(doc)
            paragraph_docs.append(doc)
        
        # 创建文档级内容（合并同一来源的段落）
        document_docs = []
        for source, docs in document_groups.items():
            combined_content = "\n\n".join([doc.page_content for doc in docs])
            # 限制文档级内容长度
            if len(combined_content) > 3000:
                combined_content = combined_content[:3000] + "..."
            
            doc_metadata = docs[0].metadata.copy()
            doc_metadata['granularity'] = 'document'
            doc_metadata['chunk_count'] = len(docs)
            
            document_docs.append(Document(
                page_content=combined_content,
                metadata=doc_metadata
            ))
        
        return paragraph_docs, document_docs
    
    def _initialize_retrievers(self):
        """初始化所有检索器"""
        try:
            with st.spinner("正在初始化混合检索系统..."):
                # 段落级检索器
                self._build_vector_index(self.paragraph_docs, is_document_level=False)
                self._build_bm25_index(self.paragraph_docs, is_document_level=False)
                
                # 文档级检索器
                if self.document_docs:
                    self._build_vector_index(self.document_docs, is_document_level=True)
                    self._build_bm25_index(self.document_docs, is_document_level=True)
                
                st.success(f"混合检索系统初始化完成 - 段落数：{len(self.paragraph_docs)}, 文档数：{len(self.document_docs)}")
        except Exception as e:
            st.error(f"初始化混合检索系统失败: {str(e)}")
    
    def _build_vector_index(self, docs: List[Document], is_document_level: bool = False):
        """构建向量索引"""
        texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings.embed_documents(texts)
        
        if embeddings:
            dimension = len(embeddings[0])
            index = faiss.IndexFlatIP(dimension)
            index.add(np.array(embeddings, dtype=np.float32))
            
            if is_document_level:
                self.doc_vector_index = index
            else:
                self.vector_index = index
    
    def _build_bm25_index(self, docs: List[Document], is_document_level: bool = False):
        """构建BM25索引"""
        # 使用jieba分词
        tokenized_docs = []
        for doc in docs:
            tokens = list(jieba.cut(doc.page_content))
            tokenized_docs.append(tokens)
        
        bm25 = BM25Okapi(tokenized_docs)
        
        if is_document_level:
            self.doc_bm25_model = bm25
        else:
            self.bm25_model = bm25
    
    def retrieve(self, query: str, alpha: float = 0.7) -> List[Document]:
        """
        混合检索：结合向量检索和BM25
        alpha: 向量检索权重 (1-alpha为BM25权重)
        """
        if not self.vector_index or not self.bm25_model:
            return []
        
        try:
            # 段落级检索
            paragraph_results = self._hybrid_search(
                query, self.paragraph_docs, self.vector_index, 
                self.bm25_model, alpha, k=self.top_k // 2
            )
            
            # 文档级检索
            document_results = []
            if self.doc_vector_index and self.doc_bm25_model:
                document_results = self._hybrid_search(
                    query, self.document_docs, self.doc_vector_index,
                    self.doc_bm25_model, alpha, k=self.top_k // 4
                )
            
            # 合并结果并去重
            all_results = paragraph_results + document_results
            
            # 按分数排序并返回top_k
            all_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in all_results[:self.top_k]]
            
        except Exception as e:
            st.error(f"混合检索失败: {str(e)}")
            return []
    
    def _hybrid_search(self, query: str, docs: List[Document], 
                      vector_index, bm25_model, alpha: float, k: int) -> List[Tuple[Document, float]]:
        """执行混合搜索"""
        # 向量检索
        query_embedding = self.embeddings.embed_query(query)
        if not query_embedding:
            return []
        
        vector_scores, vector_indices = vector_index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(k * 2, len(docs))
        )
        
        # BM25检索
        query_tokens = list(jieba.cut(query))
        bm25_scores = bm25_model.get_scores(query_tokens)
        
        # 归一化分数
        vector_scores = vector_scores[0]
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # 混合分数计算
        results = []
        processed_indices = set()
        
        # 处理向量检索结果
        for i, idx in enumerate(vector_indices[0]):
            if 0 <= idx < len(docs):
                hybrid_score = alpha * vector_scores[i] + (1 - alpha) * bm25_scores[idx]
                results.append((docs[idx], hybrid_score))
                processed_indices.add(idx)
        
        # 添加高分的BM25结果
        for idx, score in enumerate(bm25_scores):
            if idx not in processed_indices and score > 0.5:  # 只添加高分BM25结果
                hybrid_score = alpha * 0 + (1 - alpha) * score
                results.append((docs[idx], hybrid_score))
        
        return results