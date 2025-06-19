# hard_negative_mining.py - 困难负样本挖掘优化模块
import os
import json
import logging
import numpy as np
import random
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardNegativeMiner:
    """困难负样本挖掘器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化困难负样本挖掘器
        
        Args:
            model_path: 用于挖掘的模型路径，如果为None则使用预训练模型
        """
        self.model_path = model_path or 'BAAI/bge-large-zh-v1.5'
        self.model = None
        self.tfidf_vectorizer = None
        self.domain_keywords = self._build_domain_keywords()
        
    def _build_domain_keywords(self) -> Set[str]:
        """构建机器学习领域关键词集合"""
        keywords = {
            # 基础概念
            '机器学习', '深度学习', '人工智能', '神经网络', '算法', '模型',
            '训练', '测试', '验证', '预测', '分类', '回归', '聚类',
            
            # 具体算法
            '线性回归', '逻辑回归', '决策树', '随机森林', 'SVM', '支持向量机',
            'KNN', 'K均值', 'LSTM', 'CNN', 'RNN', 'Transformer', 'BERT',
            'XGBoost', 'LightGBM', 'AdaBoost', '朴素贝叶斯',
            
            # 技术术语
            '梯度下降', '反向传播', '过拟合', '欠拟合', '正则化', '交叉验证',
            '特征工程', '数据预处理', '标准化', '归一化', '降维', 'PCA',
            '激活函数', '损失函数', '优化器', '学习率', '批次大小',
            
            # 评估指标
            '准确率', '精确率', '召回率', 'F1', 'AUC', 'ROC', 'MSE', 'RMSE',
            'MAE', '混淆矩阵', 'R方', '相关系数',
            
            # 数据相关
            '数据集', '特征', '标签', '样本', '训练集', '测试集', '验证集',
            '数据挖掘', '数据科学', '大数据', '特征选择', '特征提取'
        }
        return keywords
    
    def load_model(self):
        """加载用于挖掘的模型"""
        if self.model is None:
            logger.info(f"加载模型用于困难负样本挖掘: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def build_semantic_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                    all_texts: List[str],
                                    top_k: int = 10) -> List[Tuple[str, str]]:
        """
        基于语义相似度的困难负样本挖掘
        
        Args:
            positive_pairs: 正样本对列表
            all_texts: 所有可用文本
            top_k: 每个查询返回的困难负样本数量
        """
        logger.info("开始基于语义相似度的困难负样本挖掘...")
        
        self.load_model()
        
        hard_negatives = []
        positive_set = self._build_positive_set(positive_pairs)
        
        # 编码所有文本
        logger.info("编码文本...")
        all_embeddings = self.model.encode(all_texts, batch_size=32, show_progress_bar=True)
        text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}
        
        # 为每个正样本对挖掘困难负样本
        processed = 0
        for query, positive_doc in positive_pairs[:100]:  # 限制数量避免过长时间
            if processed % 20 == 0:
                logger.info(f"处理进度: {processed}/{min(100, len(positive_pairs))}")
            
            query_embedding = text_to_embedding[query]
            
            # 计算与所有文档的相似度
            similarities = []
            candidate_docs = []
            
            for doc in all_texts:
                if doc != query and doc != positive_doc and (query, doc) not in positive_set:
                    doc_embedding = text_to_embedding[doc]
                    sim = cosine_similarity([query_embedding], [doc_embedding])[0, 0]
                    similarities.append(sim)
                    candidate_docs.append(doc)
            
            # 选择相似度高但不是正样本的文档作为困难负样本
            if similarities:
                # 排序并选择top-k个最相似的非正样本
                sorted_indices = np.argsort(similarities)[::-1]
                
                for i in range(min(top_k // 4, len(sorted_indices))):  # 每个查询取少量困难负样本
                    idx = sorted_indices[i]
                    hard_negatives.append((query, candidate_docs[idx]))
            
            processed += 1
        
        logger.info(f"基于语义相似度挖掘到 {len(hard_negatives)} 个困难负样本")
        return hard_negatives
    
    def build_keyword_based_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                         all_texts: List[str]) -> List[Tuple[str, str]]:
        """
        基于关键词重叠的困难负样本构建
        
        策略：选择与查询有一定关键词重叠但语义不同的文档
        """
        logger.info("开始基于关键词重叠的困难负样本挖掘...")
        
        hard_negatives = []
        positive_set = self._build_positive_set(positive_pairs)
        
        # 为所有文本提取关键词
        text_keywords = {}
        for text in all_texts:
            keywords = self._extract_keywords(text)
            text_keywords[text] = keywords
        
        for query, positive_doc in positive_pairs[:150]:  # 处理更多样本
            query_keywords = text_keywords[query]
            
            # 找到有一定关键词重叠但不是正样本的文档
            candidates = []
            
            for doc in all_texts:
                if doc != query and doc != positive_doc and (query, doc) not in positive_set:
                    doc_keywords = text_keywords[doc]
                    
                    # 计算关键词重叠度
                    overlap = len(query_keywords.intersection(doc_keywords))
                    total_keywords = len(query_keywords.union(doc_keywords))
                    
                    if total_keywords > 0:
                        overlap_ratio = overlap / total_keywords
                        
                        # 选择有一定重叠但不太高的文档（0.1-0.4之间）
                        if 0.1 <= overlap_ratio <= 0.4:
                            candidates.append((doc, overlap_ratio))
            
            # 按重叠度排序，选择中等重叠度的作为困难负样本
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for doc, ratio in candidates[:2]:  # 每个查询最多取2个
                hard_negatives.append((query, doc))
        
        logger.info(f"基于关键词重叠挖掘到 {len(hard_negatives)} 个困难负样本")
        return hard_negatives
    
    def build_tfidf_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                 all_texts: List[str]) -> List[Tuple[str, str]]:
        """
        基于TF-IDF相似度的困难负样本构建
        
        策略：使用TF-IDF找到词汇相似但语义可能不同的文档
        """
        logger.info("开始基于TF-IDF的困难负样本挖掘...")
        
        # 构建TF-IDF向量化器
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None  # 保留停用词，因为我们处理的是中文
            )
            self.tfidf_vectorizer.fit(all_texts)
        
        # 计算所有文本的TF-IDF向量
        tfidf_matrix = self.tfidf_vectorizer.transform(all_texts)
        text_to_index = {text: i for i, text in enumerate(all_texts)}
        
        hard_negatives = []
        positive_set = self._build_positive_set(positive_pairs)
        
        for query, positive_doc in positive_pairs[:100]:
            if query not in text_to_index:
                continue
                
            query_idx = text_to_index[query]
            query_vector = tfidf_matrix[query_idx]
            
            # 计算与所有文档的TF-IDF相似度
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            # 找到TF-IDF相似度中等的文档作为困难负样本
            candidates = []
            for i, sim in enumerate(similarities):
                doc = all_texts[i]
                if (doc != query and doc != positive_doc and 
                    (query, doc) not in positive_set and
                    0.2 <= sim <= 0.6):  # 中等相似度范围
                    candidates.append((doc, sim))
            
            # 按相似度排序，选择中等相似度的
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for doc, sim in candidates[:3]:  # 每个查询最多取3个
                hard_negatives.append((query, doc))
        
        logger.info(f"基于TF-IDF挖掘到 {len(hard_negatives)} 个困难负样本")
        return hard_negatives
    
    def build_cross_domain_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                        all_texts: List[str]) -> List[Tuple[str, str]]:
        """
        构建跨子领域的困难负样本
        
        策略：选择同属机器学习但不同子领域的文本对
        """
        logger.info("开始构建跨子领域困难负样本...")
        
        # 定义机器学习子领域
        subdomains = {
            'deep_learning': ['深度学习', '神经网络', 'LSTM', 'CNN', 'RNN', 'Transformer', 'BERT'],
            'traditional_ml': ['线性回归', '逻辑回归', '决策树', '随机森林', 'SVM', 'KNN'],
            'data_processing': ['数据预处理', '特征工程', '标准化', '归一化', '降维', 'PCA'],
            'evaluation': ['准确率', '精确率', '召回率', 'F1', 'AUC', 'ROC', '交叉验证'],
            'optimization': ['梯度下降', '优化器', '学习率', '正则化', '过拟合', '欠拟合']
        }
        
        # 为每个文本分类子领域
        text_domains = {}
        for text in all_texts:
            text_domains[text] = self._classify_subdomain(text, subdomains)
        
        hard_negatives = []
        positive_set = self._build_positive_set(positive_pairs)
        
        for query, positive_doc in positive_pairs:
            query_domain = text_domains[query]
            
            # 找到不同子领域但仍属于机器学习的文档
            for doc in all_texts:
                if (doc != query and doc != positive_doc and 
                    (query, doc) not in positive_set):
                    
                    doc_domain = text_domains[doc]
                    
                    # 如果属于不同子领域且都包含机器学习关键词
                    if (query_domain != doc_domain and 
                        query_domain != 'unknown' and doc_domain != 'unknown' and
                        self._contains_ml_keywords(doc)):
                        hard_negatives.append((query, doc))
                        
                        # 限制每个查询的困难负样本数量
                        query_negatives = [hn for hn in hard_negatives if hn[0] == query]
                        if len(query_negatives) >= 2:
                            break
        
        logger.info(f"构建跨子领域困难负样本 {len(hard_negatives)} 个")
        return hard_negatives
    
    def build_length_similar_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                          all_texts: List[str]) -> List[Tuple[str, str]]:
        """
        构建长度相似的困难负样本
        
        策略：选择长度相近但内容不相关的文本
        """
        logger.info("开始构建长度相似的困难负样本...")
        
        hard_negatives = []
        positive_set = self._build_positive_set(positive_pairs)
        
        for query, positive_doc in positive_pairs:
            query_len = len(query)
            
            # 找到长度相似的文档
            candidates = []
            for doc in all_texts:
                if (doc != query and doc != positive_doc and 
                    (query, doc) not in positive_set):
                    
                    doc_len = len(doc)
                    len_ratio = min(query_len, doc_len) / max(query_len, doc_len)
                    
                    # 长度相似度在0.7-1.0之间
                    if len_ratio >= 0.7:
                        # 但内容相关性较低
                        keyword_overlap = self._calculate_keyword_overlap(query, doc)
                        if keyword_overlap < 0.3:  # 关键词重叠度较低
                            candidates.append(doc)
            
            # 随机选择一些候选
            if candidates:
                selected = random.sample(candidates, min(2, len(candidates)))
                for doc in selected:
                    hard_negatives.append((query, doc))
        
        logger.info(f"构建长度相似困难负样本 {len(hard_negatives)} 个")
        return hard_negatives
    
    def build_comprehensive_hard_negatives(self, positive_pairs: List[Tuple[str, str]], 
                                         all_texts: List[str],
                                         strategies: List[str] = None) -> List[Tuple[str, str]]:
        """
        综合多种策略构建困难负样本
        
        Args:
            positive_pairs: 正样本对
            all_texts: 所有可用文本
            strategies: 要使用的策略列表
        """
        if strategies is None:
            strategies = ['semantic', 'keyword', 'tfidf', 'cross_domain', 'length_similar']
        
        logger.info(f"使用策略构建困难负样本: {strategies}")
        
        all_hard_negatives = []
        
        # 基于语义相似度的困难负样本
        if 'semantic' in strategies:
            semantic_negatives = self.build_semantic_hard_negatives(positive_pairs, all_texts)
            all_hard_negatives.extend(semantic_negatives)
        
        # 基于关键词重叠的困难负样本
        if 'keyword' in strategies:
            keyword_negatives = self.build_keyword_based_hard_negatives(positive_pairs, all_texts)
            all_hard_negatives.extend(keyword_negatives)
        
        # 基于TF-IDF的困难负样本
        if 'tfidf' in strategies:
            tfidf_negatives = self.build_tfidf_hard_negatives(positive_pairs, all_texts)
            all_hard_negatives.extend(tfidf_negatives)
        
        # 跨子领域困难负样本
        if 'cross_domain' in strategies:
            cross_domain_negatives = self.build_cross_domain_hard_negatives(positive_pairs, all_texts)
            all_hard_negatives.extend(cross_domain_negatives)
        
        # 长度相似困难负样本
        if 'length_similar' in strategies:
            length_negatives = self.build_length_similar_hard_negatives(positive_pairs, all_texts)
            all_hard_negatives.extend(length_negatives)
        
        # 去重
        unique_negatives = list(set(all_hard_negatives))
        
        logger.info(f"总计生成 {len(all_hard_negatives)} 个困难负样本")
        logger.info(f"去重后保留 {len(unique_negatives)} 个困难负样本")
        
        # 按策略统计
        strategy_counts = {}
        if 'semantic' in strategies:
            strategy_counts['semantic'] = len(semantic_negatives)
        if 'keyword' in strategies:
            strategy_counts['keyword'] = len(keyword_negatives)
        if 'tfidf' in strategies:
            strategy_counts['tfidf'] = len(tfidf_negatives)
        if 'cross_domain' in strategies:
            strategy_counts['cross_domain'] = len(cross_domain_negatives)
        if 'length_similar' in strategies:
            strategy_counts['length_similar'] = len(length_negatives)
        
        logger.info(f"各策略生成数量: {strategy_counts}")
        
        return unique_negatives
    
    def _build_positive_set(self, positive_pairs: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """构建正样本对的集合（双向）"""
        positive_set = set()
        for text1, text2 in positive_pairs:
            positive_set.add((text1, text2))
            positive_set.add((text2, text1))
        return positive_set
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取文本关键词"""
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤长度和领域关键词
        keywords = set()
        for word in words:
            if len(word) >= 2 and word in self.domain_keywords:
                keywords.add(word)
        
        return keywords
    
    def _classify_subdomain(self, text: str, subdomains: Dict[str, List[str]]) -> str:
        """分类文本所属的子领域"""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in subdomains.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            domain_scores[domain] = score
        
        # 返回得分最高的领域，如果都是0则返回unknown
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else 'unknown'
    
    def _contains_ml_keywords(self, text: str) -> bool:
        """检查文本是否包含机器学习关键词"""
        text_keywords = self._extract_keywords(text)
        return len(text_keywords) > 0
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的关键词重叠度"""
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if len(keywords1) == 0 and len(keywords2) == 0:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def save_hard_negatives_analysis(self, hard_negatives: List[Tuple[str, str]], 
                                   output_file: str = 'outputs/data/hard_negatives_analysis.json'):
        """保存困难负样本分析结果"""
        
        analysis = {
            'total_hard_negatives': len(hard_negatives),
            'sample_pairs': hard_negatives[:10],  # 保存前10个样本
            'statistics': {
                'avg_query_length': np.mean([len(pair[0]) for pair in hard_negatives]),
                'avg_doc_length': np.mean([len(pair[1]) for pair in hard_negatives]),
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"困难负样本分析已保存到: {output_file}")


# 使用示例和测试函数
def test_hard_negative_mining():
    """测试困难负样本挖掘功能"""
    
    # 模拟一些正样本对和文本数据
    positive_pairs = [
        ("什么是机器学习？", "机器学习是人工智能的一个分支..."),
        ("深度学习的原理", "深度学习使用多层神经网络..."),
        ("如何进行特征工程？", "特征工程包括特征选择和特征提取...")
    ]
    
    all_texts = [
        "什么是机器学习？",
        "机器学习是人工智能的一个分支...",
        "深度学习的原理", 
        "深度学习使用多层神经网络...",
        "如何进行特征工程？",
        "特征工程包括特征选择和特征提取...",
        "Python编程基础教程",
        "数据库设计原理",
        "网页前端开发技术",
        "决策树算法详解",
        "随机森林在分类中的应用",
        "支持向量机的数学原理"
    ]
    
    # 创建困难负样本挖掘器
    miner = HardNegativeMiner()
    
    # 测试不同策略
    strategies = ['keyword', 'tfidf', 'cross_domain', 'length_similar']
    hard_negatives = miner.build_comprehensive_hard_negatives(
        positive_pairs, all_texts, strategies
    )
    
    print(f"生成困难负样本数量: {len(hard_negatives)}")
    for i, (query, doc) in enumerate(hard_negatives[:5]):
        print(f"{i+1}. Query: {query[:50]}...")
        print(f"   Doc: {doc[:50]}...")
        print()

if __name__ == "__main__":
    test_hard_negative_mining()