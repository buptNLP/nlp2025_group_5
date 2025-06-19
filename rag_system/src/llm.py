from typing import List, Dict
from openai import OpenAI
from langchain.schema import Document
import streamlit as st
from src.config import API_BASE, API_KEY, MODEL, ML_ASSISTANT_ROLE, SYSTEM_PROMPT
import time
from src.document_loader import DocumentLoader
import jieba
import jieba.analyse

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
        
    def _is_ml_related(self, query: str) -> bool:
        """判断问题是否与机器学习相关"""
        ml_keywords = [
            '机器学习', '深度学习', '神经网络', '算法', '模型', '训练', '预测',
            '分类', '回归', '聚类', '特征', '数据', 'AI', '人工智能',
            'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy',
            '监督学习', '无监督学习', '强化学习', '过拟合', '欠拟合',
            'cnn', 'rnn', 'lstm', 'transformer', 'bert', 'gpt'
        ]
        
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in ml_keywords)
    
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
    2. 回答中应提取和整合参考资料中的核心概念、定义和方法
    3. 使用明确的引用格式，例如"根据[参考资料1]，贝叶斯模型的核心原理是..."
    4. 回答的内容应该有深度和专业性，不要只是泛泛而谈
    5. 在适当位置引用参考资料中的具体术语、公式或定义
    6. 回答的结构应该清晰，可以包含：概念解释、工作原理、应用场景、优缺点等
    7. 在回答的结尾处，以"参考文献："开头列出所有引用的参考资料

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
                        temperature=0.5,  # 降低温度以提高准确性和对指令的遵循
                        max_tokens=2000   # 增加tokens以允许更详细的回答
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
3. 回答时可以引用原文的关键句子，标明来源（如：片段1）
4. 回答要简洁、专业、有条理，适合机器学习领域的用户阅读
5. 提供完整、系统的回答，而不是简单复述文档片段
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
        """优化的关键词提取和文章匹配方法"""
        try:
            # 导入需要的模块
            from src.document_loader import DocumentLoader
            import jieba.analyse
            
            # 获取所有文章
            loader = DocumentLoader()
            all_articles = loader.get_knowledge_base_articles()
            
            if not all_articles:
                return []
            
            # 定义机器学习相关的重要关键词权重加强
            ml_keywords = {
                '机器学习': 3, '深度学习': 3, '神经网络': 3, '卷积': 3, '循环': 3, 
                '强化学习': 3, '分类': 2, '回归': 2, '聚类': 2, '监督': 2, '无监督': 2,
                '模型': 2, '算法': 2, '特征': 2, '优化': 2, '参数': 2, '损失函数': 3,
                '过拟合': 3, '欠拟合': 3, '贝叶斯': 3, '决策树': 3, '随机森林': 3,
                'SVM': 3, 'LSTM': 3, 'CNN': 3, 'RNN': 3, 'GAN': 3, 'Transformer': 3
            }
            
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