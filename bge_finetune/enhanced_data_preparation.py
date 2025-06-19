# enhanced_data_preparation.py - 集成困难负样本挖掘的增强版数据准备
import os
import json
import random
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import pandas as pd

# 导入困难负样本挖掘模块
from hard_negative_mining import HardNegativeMiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataPreparation:
    """增强版数据准备，集成困难负样本挖掘"""
    
    def __init__(self, data_config: Dict, hard_negative_config: Dict = None):
        self.data_config = data_config
        self.hard_negative_config = hard_negative_config or self._default_hard_negative_config()
        self.articles = []
        self.qa_pairs = []
        self.hard_negative_miner = None
        self.setup_directories()
        
    def _default_hard_negative_config(self) -> Dict:
        """默认困难负样本配置"""
        return {
            'enabled': True,
            'strategies': ['semantic', 'keyword', 'tfidf', 'cross_domain'],
            'ratio_to_positive': 0.8,  # 困难负样本占正样本的比例
            'random_negative_ratio': 0.3,  # 随机负样本占总负样本的比例
            'use_pretrained_model': True,  # 是否使用预训练模型进行挖掘
            'model_path': 'models/bge-large-zh-v1.5'
        }
    
    def setup_directories(self):
        """创建必要的目录"""
        directories = ['outputs', 'outputs/logs', 'outputs/data', 'outputs/analysis']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """加载原始数据"""
        logger.info("开始加载数据...")
        
        # 加载文章数据
        articles_file = self.data_config['articles_file']
        with open(articles_file, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)
        
        # 加载文章正文内容
        txt_dir = self.data_config['article_texts_dir']
        for article in self.articles:
            article_id = article['article_id']
            txt_file = os.path.join(txt_dir, f"{article_id}.txt")
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 智能截断：保持段落完整性
                    article['full_content'] = self._smart_truncate(content, 2000)
            else:
                article['full_content'] = article.get('content_preview', '')
        
        # 加载QA对数据
        qa_file = self.data_config['qa_pairs_file']
        with open(qa_file, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        
        logger.info(f"数据加载完成: {len(self.articles)} 篇文章, {len(self.qa_pairs)} 条QA对")
        
        # 数据质量检查和清洗
        self._clean_and_validate_data()
        
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """智能截断文本，保持段落完整性"""
        if len(text) <= max_length:
            return text
        
        # 尝试在段落边界截断
        paragraphs = text.split('\n\n')
        truncated = ""
        
        for para in paragraphs:
            if len(truncated + para) <= max_length:
                truncated += para + '\n\n'
            else:
                break
        
        # 如果没有找到合适的段落边界，直接截断
        if len(truncated.strip()) < max_length * 0.5:
            truncated = text[:max_length]
            # 尝试在句号处截断
            last_period = truncated.rfind('。')
            if last_period > max_length * 0.7:
                truncated = truncated[:last_period + 1]
        
        return truncated.strip()
    
    def _clean_and_validate_data(self):
        """清洗和验证数据质量"""
        logger.info("开始数据清洗和质量验证...")
        
        # 清洗QA对
        original_qa_count = len(self.qa_pairs)
        self.qa_pairs = [
            qa for qa in self.qa_pairs
            if (len(qa.get('question', '').strip()) >= 5 and 
                len(qa.get('answer', '').strip()) >= 10 and
                len(qa.get('answer', '')) <= 2000 and
                len(qa.get('question', '')) <= 500)
        ]
        
        # 清洗文章数据
        original_article_count = len(self.articles)
        self.articles = [
            article for article in self.articles
            if (len(article.get('title', '').strip()) >= 5 and 
                len(article.get('full_content', '').strip()) >= 20)
        ]
        
        logger.info(f"数据清洗完成: QA对 {original_qa_count} -> {len(self.qa_pairs)}, "
                   f"文章 {original_article_count} -> {len(self.articles)}")
        
        # 去重处理
        self._remove_duplicates()
    
    def _remove_duplicates(self):
        """去除重复数据"""
        # QA对去重
        qa_texts = set()
        unique_qa_pairs = []
        
        for qa in self.qa_pairs:
            qa_key = (qa['question'].strip(), qa['answer'].strip())
            if qa_key not in qa_texts:
                qa_texts.add(qa_key)
                unique_qa_pairs.append(qa)
        
        # 文章去重
        article_titles = set()
        unique_articles = []
        
        for article in self.articles:
            title = article['title'].strip()
            if title not in article_titles:
                article_titles.add(title)
                unique_articles.append(article)
        
        original_qa = len(self.qa_pairs)
        original_articles = len(self.articles)
        
        self.qa_pairs = unique_qa_pairs
        self.articles = unique_articles
        
        logger.info(f"去重完成: QA对 {original_qa} -> {len(self.qa_pairs)}, "
                   f"文章 {original_articles} -> {len(self.articles)}")
    
    def build_positive_pairs(self) -> List[Tuple[str, str]]:
        """构建正样本对"""
        logger.info("构建正样本对...")
        
        positive_pairs = []
        
        # 1. 基于QA对构建
        qa_pairs_count = 0
        for qa in self.qa_pairs:
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            
            if question and answer:
                positive_pairs.append((question, answer))
                qa_pairs_count += 1
        
        logger.info(f"从QA对构建正样本: {qa_pairs_count} 对")
        
        # 2. 基于文档内容构建
        doc_pairs_count = 0
        for article in self.articles:
            title = article.get('title', '').strip()
            content_preview = article.get('content_preview', '').strip()
            full_content = article.get('full_content', '').strip()
            
            # 标题-内容预览对
            if title and content_preview and len(content_preview) >= 50:
                positive_pairs.append((title, content_preview))
                doc_pairs_count += 1
            
            # 标题-完整内容对（取前500字符）
            if title and full_content and full_content != content_preview:
                content_snippet = full_content[:500] if len(full_content) > 500 else full_content
                if len(content_snippet) >= 100:
                    positive_pairs.append((title, content_snippet))
                    doc_pairs_count += 1
            
            # 内容预览-完整内容片段对（如果差异较大）
            if (content_preview and full_content and 
                len(full_content) > len(content_preview) * 1.5):
                # 取完整内容的中间部分
                start_idx = len(content_preview)
                end_idx = min(start_idx + 400, len(full_content))
                content_middle = full_content[start_idx:end_idx]
                
                if len(content_middle) >= 100:
                    positive_pairs.append((content_preview, content_middle))
                    doc_pairs_count += 1
        
        logger.info(f"从文档构建正样本: {doc_pairs_count} 对")
        logger.info(f"总正样本对数: {len(positive_pairs)}")
        
        return positive_pairs
    
    def initialize_hard_negative_miner(self):
        """初始化困难负样本挖掘器"""
        if self.hard_negative_config['enabled'] and self.hard_negative_miner is None:
            model_path = (self.hard_negative_config['model_path'] 
                         if self.hard_negative_config['use_pretrained_model'] 
                         else None)
            
            self.hard_negative_miner = HardNegativeMiner(model_path)
            logger.info("困难负样本挖掘器初始化完成")
    
    def build_enhanced_negative_pairs(self, positive_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """构建增强的负样本对（包含困难负样本）"""
        logger.info("开始构建增强负样本对...")
        
        # 收集所有文本
        all_texts = []
        for pair in positive_pairs:
            all_texts.extend(pair)
        all_texts = list(set(all_texts))  # 去重
        
        negative_pairs = []
        target_negative_count = int(len(positive_pairs) * 0.7)  # 总负样本数量
        
        if self.hard_negative_config['enabled']:
            # 使用困难负样本挖掘
            self.initialize_hard_negative_miner()
            
            hard_negative_count = int(target_negative_count * (1 - self.hard_negative_config['random_negative_ratio']))
            random_negative_count = target_negative_count - hard_negative_count
            
            logger.info(f"目标负样本分布: 困难负样本 {hard_negative_count}, 随机负样本 {random_negative_count}")
            
            # 1. 构建困难负样本
            try:
                hard_negatives = self.hard_negative_miner.build_comprehensive_hard_negatives(
                    positive_pairs=positive_pairs,
                    all_texts=all_texts,
                    strategies=self.hard_negative_config['strategies']
                )
                
                # 限制困难负样本数量
                if len(hard_negatives) > hard_negative_count:
                    hard_negatives = random.sample(hard_negatives, hard_negative_count)
                
                negative_pairs.extend(hard_negatives)
                
                logger.info(f"实际生成困难负样本: {len(hard_negatives)} 个")
                
                # 保存困难负样本分析
                self.hard_negative_miner.save_hard_negatives_analysis(
                    hard_negatives, 'outputs/analysis/hard_negatives_analysis.json'
                )
                
            except Exception as e:
                logger.warning(f"困难负样本挖掘失败，将使用随机负样本: {e}")
                random_negative_count = target_negative_count
            
            # 2. 补充随机负样本
            remaining_count = target_negative_count - len(negative_pairs)
            if remaining_count > 0:
                random_negatives = self._build_random_negatives(
                    positive_pairs, all_texts, remaining_count
                )
                negative_pairs.extend(random_negatives)
        
        else:
            # 仅使用随机负样本
            logger.info("困难负样本挖掘已禁用，使用随机负样本")
            random_negatives = self._build_random_negatives(
                positive_pairs, all_texts, target_negative_count
            )
            negative_pairs = random_negatives
        
        logger.info(f"负样本构建完成: 总计 {len(negative_pairs)} 个")
        return negative_pairs
    
    def _build_random_negatives(self, positive_pairs: List[Tuple[str, str]], 
                               all_texts: List[str], count: int) -> List[Tuple[str, str]]:
        """构建随机负样本"""
        positive_set = set()
        for text1, text2 in positive_pairs:
            positive_set.add((text1, text2))
            positive_set.add((text2, text1))
        
        random_negatives = []
        attempts = 0
        max_attempts = count * 20
        
        while len(random_negatives) < count and attempts < max_attempts:
            attempts += 1
            
            if len(all_texts) < 2:
                break
            
            text1, text2 = random.sample(all_texts, 2)
            
            if (text1, text2) not in positive_set:
                random_negatives.append((text1, text2))
        
        return random_negatives
    
    def create_input_examples(self) -> List[InputExample]:
        """创建InputExample格式的训练样本"""
        logger.info("开始创建训练样本...")
        
        # 构建正样本对
        positive_pairs = self.build_positive_pairs()
        
        # 构建增强负样本对
        negative_pairs = self.build_enhanced_negative_pairs(positive_pairs)
        
        # 转换为InputExample格式
        examples = []
        
        # 添加正样本
        for text1, text2 in positive_pairs:
            examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # 添加负样本
        for text1, text2 in negative_pairs:
            examples.append(InputExample(texts=[text1, text2], label=0.0))
        
        # 样本分析
        positive_count = len(positive_pairs)
        negative_count = len(negative_pairs)
        total_count = len(examples)
        
        logger.info(f"样本创建完成:")
        logger.info(f"  正样本: {positive_count} ({positive_count/total_count*100:.1f}%)")
        logger.info(f"  负样本: {negative_count} ({negative_count/total_count*100:.1f}%)")
        logger.info(f"  总计: {total_count} 个样本")
        
        # 保存样本分析
        self._save_sample_analysis(positive_pairs, negative_pairs)
        
        return examples
    
    def _save_sample_analysis(self, positive_pairs: List[Tuple[str, str]], 
                             negative_pairs: List[Tuple[str, str]]):
        """保存样本分析结果"""
        
        def analyze_pairs(pairs, label):
            if not pairs:
                return {}
            
            lengths = [len(text1) + len(text2) for text1, text2 in pairs]
            query_lengths = [len(text1) for text1, text2 in pairs]
            doc_lengths = [len(text2) for text1, text2 in pairs]
            
            return {
                'count': len(pairs),
                'avg_total_length': sum(lengths) / len(lengths),
                'avg_query_length': sum(query_lengths) / len(query_lengths),
                'avg_doc_length': sum(doc_lengths) / len(doc_lengths),
                'min_total_length': min(lengths),
                'max_total_length': max(lengths),
                'samples': pairs[:5]  # 保存前5个样本
            }
        
        analysis = {
            'positive_samples': analyze_pairs(positive_pairs, 'positive'),
            'negative_samples': analyze_pairs(negative_pairs, 'negative'),
            'hard_negative_config': self.hard_negative_config,
            'data_sources': {
                'articles_count': len(self.articles),
                'qa_pairs_count': len(self.qa_pairs)
            }
        }
        
        analysis_file = 'outputs/analysis/sample_analysis.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"样本分析已保存到: {analysis_file}")
    
    def split_dataset(self, examples: List[InputExample], 
                     test_size: float = 0.3, 
                     eval_size: float = 0.5) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
        """划分训练集、验证集、测试集"""
        
        # 分层抽样，保持正负样本比例
        train_examples, temp_examples = train_test_split(
            examples, test_size=test_size, random_state=42, 
            stratify=[ex.label for ex in examples]
        )
        
        eval_examples, test_examples = train_test_split(
            temp_examples, test_size=eval_size, random_state=42,
            stratify=[ex.label for ex in temp_examples]
        )
        
        # 检查各集合的正负样本比例
        def check_balance(dataset, name):
            positive = sum(1 for ex in dataset if ex.label > 0.5)
            total = len(dataset)
            ratio = positive / total if total > 0 else 0
            logger.info(f"  {name}: {total} 样本, 正样本比例: {ratio:.3f}")
        
        logger.info("数据集划分完成:")
        check_balance(train_examples, "训练集")
        check_balance(eval_examples, "验证集")
        check_balance(test_examples, "测试集")
        
        return train_examples, eval_examples, test_examples
    
    def save_processed_data(self, train_examples: List[InputExample], 
                          eval_examples: List[InputExample], 
                          test_examples: List[InputExample]):
        """保存处理后的数据"""
        
        def examples_to_dict(examples):
            return [
                {
                    'text1': ex.texts[0],
                    'text2': ex.texts[1], 
                    'label': float(ex.label)
                }
                for ex in examples
            ]
        
        datasets = {
            'train': examples_to_dict(train_examples),
            'eval': examples_to_dict(eval_examples),
            'test': examples_to_dict(test_examples),
            'metadata': {
                'hard_negative_config': self.hard_negative_config,
                'creation_timestamp': json.dumps(str(pd.Timestamp.now())),
                'total_samples': len(train_examples) + len(eval_examples) + len(test_examples)
            }
        }
        
        output_file = 'outputs/data/enhanced_processed_datasets.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"增强处理数据已保存到: {output_file}")
        
        # 保存详细统计
        stats = {
            'dataset_splits': {
                'train': len(train_examples),
                'eval': len(eval_examples), 
                'test': len(test_examples)
            },
            'sample_composition': {
                'total_positive': sum(1 for ex in train_examples + eval_examples + test_examples if ex.label > 0.5),
                'total_negative': sum(1 for ex in train_examples + eval_examples + test_examples if ex.label <= 0.5),
                'train_positive': sum(1 for ex in train_examples if ex.label > 0.5),
                'train_negative': sum(1 for ex in train_examples if ex.label <= 0.5)
            },
            'hard_negative_enabled': self.hard_negative_config['enabled'],
            'strategies_used': self.hard_negative_config['strategies'] if self.hard_negative_config['enabled'] else []
        }
        
        stats_file = 'outputs/analysis/enhanced_data_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return output_file, stats_file
    
    def run_enhanced_data_preparation(self) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
        """运行增强版数据准备流程"""
        logger.info("="*60)
        logger.info("开始增强版数据准备流程")
        logger.info("="*60)
        
        # 1. 加载和清洗数据
        self.load_data()
        
        # 2. 创建训练样本（包含困难负样本）
        examples = self.create_input_examples()
        
        # 3. 划分数据集
        train_examples, eval_examples, test_examples = self.split_dataset(examples)
        
        # 4. 保存处理后的数据
        self.save_processed_data(train_examples, eval_examples, test_examples)
        
        logger.info("增强版数据准备流程完成")
        logger.info("="*60)
        
        return train_examples, eval_examples, test_examples

def main():
    """主函数 - 运行增强版数据准备"""
    
    # 导入pandas用于时间戳
    import pandas as pd
    
    # 设置随机种子
    random.seed(42)
    
    # 数据配置
    data_config = {
        'articles_file': 'data/csdn_articles_filtered.json',
        'qa_pairs_file': 'data/filtered_qa_pairs.json',
        'article_texts_dir': 'data/article_texts/'
    }
    
    # 困难负样本配置
    hard_negative_config = {
        'enabled': True,
        'strategies': ['semantic', 'keyword', 'tfidf', 'cross_domain'],
        'ratio_to_positive': 0.8,
        'random_negative_ratio': 0.3,
        'use_pretrained_model': True,
        'model_path': 'models/bge-large-zh-v1.5'
    }
    
    # 检查必要文件
    required_files = [
        data_config['articles_file'],
        data_config['qa_pairs_file']
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"缺少必要文件: {file_path}")
            return
    
    logger.info("困难负样本配置:")
    for key, value in hard_negative_config.items():
        logger.info(f"  {key}: {value}")
    
    # 运行增强版数据准备
    enhanced_prep = EnhancedDataPreparation(data_config, hard_negative_config)
    train_examples, eval_examples, test_examples = enhanced_prep.run_enhanced_data_preparation()
    
    print(f"\n增强版数据准备完成!")
    print(f"训练集: {len(train_examples)} 样本")
    print(f"验证集: {len(eval_examples)} 样本") 
    print(f"测试集: {len(test_examples)} 样本")
    print(f"\n查看详细分析: outputs/analysis/")
    print(f"下一步运行: python model_training.py")

if __name__ == "__main__":
    main()