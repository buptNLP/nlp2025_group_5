from typing import List, Dict, Set
from openai import OpenAI
from langchain.schema import Document
import streamlit as st
from src.config import API_BASE, API_KEY, MODEL, ML_ASSISTANT_ROLE, SYSTEM_PROMPT
import time
from src.document_loader import DocumentLoader
import jieba
import jieba.analyse
import requests
import json
from functools import lru_cache

class LLM:
    def __init__(self):
        try:
            self.client = OpenAI(
                key=API_KEY,
                base_url=API_BASE
            )
        except Exception as e:
            st.error(f"初始化LLM客户端失败: {str(e)}")
            self.client = None
        
        # 机器学习关键词缓存
        self._ml_keywords_cache = None
    
    @lru_cache(maxsize=1)
    def _load_ml_keywords(self) -> Set[str]:
        """从Google机器学习词汇库加载关键词"""
        if self._ml_keywords_cache is not None:
            return self._ml_keywords_cache
        
        # Google机器学习词汇表URL
        urls = [
            "https://developers.google.com/machine-learning/glossary",
            "https://raw.githubusercontent.com/tensorflow/docs/master/site/en/glossary.md"
        ]
        
        ml_keywords = set()
        
        # 从在线资源获取关键词
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    # 提取机器学习相关术语
                    import re
                    
                    # 匹配常见的ML术语模式
                    patterns = [
                        r'\b(algorithm|model|neural|learning|deep|machine|ai|artificial)\s+\w+\b',
                        r'\b\w+\s+(learning|network|algorithm|model|optimization)\b',
                        r'\b(classification|regression|clustering|supervised|unsupervised|reinforcement)\b',
                        r'\b(gradient|backpropagation|convolution|activation|loss)\s+\w+\b'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        ml_keywords.update([match.strip() for match in matches if len(match.strip()) > 2])
                    
                    # 如果获取到足够的关键词就break
                    if len(ml_keywords) > 50:
                        break
                        
            except Exception as e:
                continue
        
        self._ml_keywords_cache = ml_keywords
        return ml_keywords
            
    
    def _is_ml_related(self, query: str) -> bool:
        """判断问题是否与机器学习相关"""
        try:
            ml_keywords = self._load_ml_keywords()
            query_lower = query.lower()
            
            # 检查关键词匹配
            for keyword in ml_keywords:
                if keyword.lower() in query_lower:
                    return True
            
            # 额外的语义检查 - 检查是否包含技术性词汇
            tech_indicators = ['api', 'code', 'python', 'data', 'analysis', 'predict', 'train', 'test']
            tech_count = sum(1 for indicator in tech_indicators if indicator in query_lower)
            
            # 如果包含多个技术词汇，可能是相关的
            if tech_count >= 2:
                return True
                
            return False
            
        except Exception as e:
            # 如果出错，默认认为是相关的（避免误拦截）
            return True
    
    def chat_response(self, messages: List[Dict]) -> Dict:
        """优化后的多轮对话响应，改进对参考资料的使用"""
        if not self.client:
            return {
                "content": "API连接未初始化，请检查配置",
                "references": []
            }
            
        try:
            # 检查最新的用户消息是否与机器学习相关
            latest_user_message = None
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    latest_user_message = msg['content']
                    break
            
            if latest_user_message and not self._is_ml_related(latest_user_message):
                return {
                    "content": "我是专门的机器学习助手，只能回答机器学习、深度学习、数据科学相关的问题。请问您有什么机器学习方面的问题吗？",
                    "references": []
                }
            
            # 获取相关文章
            relevant_articles = self.get_relevant_articles(latest_user_message)
            
            # 构建系统消息和提示词
            system_content = ML_ASSISTANT_ROLE
            
            # 如果有相关文章，添加到系统提示中，并改进提示词
            if relevant_articles:
                reference_texts = []
                for idx, article in enumerate(relevant_articles, 1):
                    title = article.get('title', '未知标题')
                    preview = article.get('content_preview', '无摘要')
                    link = article.get('link', '#')
                    reference_texts.append(f"[参考资料{idx}] 标题：{title}\n摘要：{preview}\n链接：{link}\n")
                
                references_prompt = "\n\n参考资料：\n" + "\n".join(reference_texts)
                
                # 更加具体的指导，要求模型深入参考资料内容
                system_content += references_prompt + """

    请按照以下要求回答用户的问题：

    1. 必须深入阅读并理解每篇参考资料的具体内容
    2. 使用明确的引用格式，例如"根据[参考资料1]，贝叶斯模型的核心原理是..."
    3. 在适当位置引用参考资料中的具体术语、公式或定义

    注意：回答必须完全基于提供的参考资料，从中提取有价值的信息，而不是生成泛泛的回答。
    """
            
            # 构建消息
            system_message = {"role": "system", "content": system_content}
            full_messages = [system_message] + messages
            
            # 添加错误处理和重试逻辑
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=MODEL,
                        messages=full_messages,
                        temperature=0.5,
                        max_tokens=2000
                    )
                    
                    return {
                        "content": response.choices[0].message.content,
                        "references": relevant_articles
                    }
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"API调用失败，正在重试... ({attempt+1}/{max_retries})")
                        time.sleep(2)  # 添加短暂延迟再重试
                    else:
                        raise
            
        except Exception as e:
            return {
                "content": f"抱歉，生成回答时出错: {str(e)}",
                "references": []
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """准备上下文信息"""
        contexts = []
        for idx, doc in enumerate(documents, 1):
            # 提取元数据信息
            source = doc.metadata.get('filename', doc.metadata.get('title', '未知来源'))
            page = doc.metadata.get('page', '')
            page_info = f" (第{page}页)" if page else ""
            
            # 格式化上下文
            context = (
                f"[片段{idx}] 来源：{source}{page_info}\n"
                f"内容：{doc.page_content.strip()}\n"
            )
            contexts.append(context)
        return "\n".join(contexts)
    
    def generate_response(self, query: str, documents: List[Document]) -> dict:
        """基于文档生成回答"""
        if not self.client:
            return {
                "answer": "API连接未初始化，请检查配置",
                "source_documents": documents
            }
            
        try:
            if not documents:
                return {
                    "answer": "抱歉，我没有找到与您问题相关的内容。请尝试修改问题或上传更多相关文档。",
                    "source_documents": []
                }
            
            # 准备上下文信息
            context = self._prepare_context(documents)
            st.info("已找到相关内容，正在生成回答...")
            
            # 构建提示词
            prompt = f"""
我将提供与用户问题相关的文档片段，请你基于这些信息回答问题。

参考资料：
{context}

用户问题：{query}

请严格遵循以下规则回答：
1. 必须严格基于参考资料回答，不要生成未在参考资料中提及的内容
2. 如果参考资料不足以完整回答问题，明确指出哪些方面的信息缺失
3. 回答要简洁、专业、有条理，适合机器学习领域的用户阅读
"""
            
            # 添加错误处理和重试逻辑
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,  # 降低温度以提高准确性
                        max_tokens=1500   # 增加最大token数以容纳更完整的回答
                    )
                    
                    answer = response.choices[0].message.content
                    
                    return {
                        "answer": answer,
                        "source_documents": documents
                    }
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"API调用失败，正在重试... ({attempt+1}/{max_retries})")
                        time.sleep(2)  # 添加短暂延迟再重试
                    else:
                        raise
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"详细错误信息: {error_details}")
            
            return {
                "answer": f"抱歉，生成回答时出错: {str(e)}",
                "source_documents": documents
            }

    def get_relevant_articles(self, query: str, top_k: int = 3) -> List[Dict]:
        """优化关键词提取和文章匹配方法"""
        try:
            # 导入需要的模块
            from src.document_loader import DocumentLoader
            import jieba.analyse
            
            # 获取所有文章
            loader = DocumentLoader()
            all_articles = loader.get_knowledge_base_articles()
            
            if not all_articles:
                return []
            
            # 获取机器学习关键词
            ml_keywords_set = self._load_ml_keywords()
            
            # 将关键词转换为权重字典
            ml_keywords = {}
            for keyword in ml_keywords_set:
                if any(term in keyword.lower() for term in ['learning', 'neural', 'deep', 'machine']):
                    ml_keywords[keyword] = 3
                elif any(term in keyword.lower() for term in ['classification', 'regression', 'clustering']):
                    ml_keywords[keyword] = 2
                else:
                    ml_keywords[keyword] = 1
            
            # 自定义jieba词典权重
            for keyword, weight in ml_keywords.items():
                jieba.suggest_freq(keyword, tune=True)
            
            # 使用jieba提取关键词，但限制数量，专注于最重要的关键词
            # 为了避免提取过多无关词，最多提取3个关键词
            keywords = jieba.analyse.extract_tags(query, topK=3, withWeight=True)
            
            # 过滤低权重的关键词，只保留权重较高的
            filtered_keywords = []
            for word, weight in keywords:
                # 如果是机器学习特定关键词，提高其权重
                if word in ml_keywords:
                    filtered_keywords.append((word, weight * ml_keywords[word]))
                elif len(word) >= 2 and weight > 0.1:  # 只保留较长且权重较高的词
                    filtered_keywords.append((word, weight))
            
            # 如果没有提取到有效关键词，尝试直接识别查询中的机器学习术语
            if not filtered_keywords:
                for ml_term, weight in ml_keywords.items():
                    if ml_term in query:
                        filtered_keywords.append((ml_term, weight))
                        break  # 只添加一个最相关的术语
            
            # 提取词语部分作为最终关键词列表
            final_keywords = [word for word, _ in filtered_keywords]
            
            # 如果还是没有关键词，使用整个查询
            if not final_keywords:
                final_keywords = [query]
            
            # 为每篇文章计算匹配分数
            scored_articles = []
            for article in all_articles:
                score = 0
                title = article.get('title', '')
                preview = article.get('content_preview', '')
                
                # 标题匹配权重显著提高
                for keyword in final_keywords:
                    if keyword in title:
                        score += 5  # 标题匹配权重大幅提高
                    if keyword in preview:
                        score += 1
                
                if score > 0:
                    scored_articles.append((article, score))
            
            # 按得分排序并返回前top_k个
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            return [article for article, _ in scored_articles[:top_k]]
            
        except Exception as e:
            st.error(f"获取相关文章时出错: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return []