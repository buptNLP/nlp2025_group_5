import json
from typing import List, Dict
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from src.config import ARTICLES_DIR, DATA_FILE, UPLOAD_DIR
import tempfile

class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；"]
        )
        
    def load_knowledge_base_documents(self) -> List[Document]:
        """加载知识库文档并进行分块"""
        documents = []
        
        try:
            # 加载元数据
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                metadata_dict = {item['article_id']: item for item in metadata_list}
            
            # 加载文章内容
            for article_path in ARTICLES_DIR.glob("*.txt"):
                article_id = article_path.stem
                if article_id in metadata_dict:
                    with open(article_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    metadata = metadata_dict[article_id]
                    chunks = self.text_splitter.create_documents(
                        texts=[text],
                        metadatas=[{
                            'article_id': article_id,
                            'title': metadata['title'],
                            'link': metadata['link'],
                            'source': 'knowledge_base'
                        }]
                    )
                    documents.extend(chunks)
            
        except Exception as e:
            st.error(f"加载知识库文档时出错: {str(e)}")
            
        return documents
    
    def load_pdf_documents(self, uploaded_files) -> List[Document]:
        """加载并处理上传的PDF文档"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # 使用临时文件处理上传文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(uploaded_file.getbuffer())
                
                st.info(f"正在处理文件: {uploaded_file.name}")
                
                # 使用更可靠的PyPDFLoader加载PDF
                loader = PyPDFLoader(temp_path)
                
                # 捕获文档加载的可能异常
                try:
                    docs = loader.load()
                    st.info(f"成功加载 {len(docs)} 页内容")
                except Exception as loader_error:
                    st.error(f"PDF加载失败: {str(loader_error)}")
                    continue
                
                if not docs:
                    st.warning(f"文件 {uploaded_file.name} 没有提取到内容")
                    continue
                
                # 对加载的文档进行分块处理
                try:
                    chunks = self.text_splitter.split_documents(docs)
                    st.info(f"文档已分割为 {len(chunks)} 个块")
                except Exception as split_error:
                    st.error(f"文档分块失败: {str(split_error)}")
                    continue
                
                # 更新元数据
                for chunk in chunks:
                    chunk.metadata.update({
                        'filename': uploaded_file.name,
                        'source': 'uploaded_pdf'
                    })
                
                documents.extend(chunks)
                
                # 清理临时文件
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
                    
            except Exception as e:
                st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
                import traceback
                st.error(f"详细错误信息: {traceback.format_exc()}")
                
        return documents
    
    def get_knowledge_base_articles(self) -> List[Dict]:
        """获取知识库文章列表"""
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"加载文章列表时出错: {str(e)}")
            return []