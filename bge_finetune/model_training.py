# model_training.py - 模型训练模块
import os
import json
import logging
import torch
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEModelTrainer:
    """BGE模型训练器"""
    
    def __init__(self, training_config: Dict):
        self.config = training_config
        self.setup_directories()
        
    def setup_directories(self):
        """创建训练相关目录"""
        directories = ['models', 'outputs/models', 'outputs/logs']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_processed_data(self) -> tuple:
        """加载预处理的数据"""
        logger.info("加载预处理数据...")
        
        data_file = 'outputs/data/enhanced_processed_datasets.json'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"未找到预处理数据文件: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        
        # 转换回InputExample格式
        def dict_to_examples(data_list):
            examples = []
            for item in data_list:
                example = InputExample(
                    texts=[item['text1'], item['text2']],
                    label=item['label']
                )
                examples.append(example)
            return examples
        
        train_examples = dict_to_examples(datasets['train'])
        eval_examples = dict_to_examples(datasets['eval'])
        test_examples = dict_to_examples(datasets['test'])
        
        logger.info(f"数据加载完成: 训练集{len(train_examples)}, "
                   f"验证集{len(eval_examples)}, 测试集{len(test_examples)}")
        
        return train_examples, eval_examples, test_examples
    
    def create_evaluator(self, eval_examples: List[InputExample]):
        """创建信息检索评估器"""
        logger.info("创建评估器...")
        
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        # 只使用正样本创建评估数据
        positive_examples = [ex for ex in eval_examples if ex.label > 0.5]
        
        for i, example in enumerate(positive_examples):
            query_id = f"q_{i}"
            doc_id = f"d_{i}"
            
            queries[query_id] = example.texts[0]
            corpus[doc_id] = example.texts[1]
            
            # 建立相关性映射
            if query_id not in relevant_docs:
                relevant_docs[query_id] = set()
            relevant_docs[query_id].add(doc_id)
        
        if not queries:
            logger.warning("没有找到正样本用于评估")
            return None
        
        logger.info(f"评估器创建完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus, 
            relevant_docs=relevant_docs,
            name="validation",
            show_progress_bar=True
        )
    
    def setup_model_and_loss(self):
        """设置模型和损失函数"""
        logger.info("初始化模型和损失函数...")
        
        # 加载预训练模型
        model_name = self.config['base_model']
        logger.info(f"加载预训练模型: {model_name}")
        
        try:
            model = SentenceTransformer(model_name)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 定义损失函数
        train_loss = losses.CosineSimilarityLoss(model)
        logger.info("损失函数设置完成")
        
        return model, train_loss
    
    def train_model(self, train_examples: List[InputExample], 
                   eval_examples: List[InputExample]) -> str:
        """训练模型"""
        logger.info("="*50)
        logger.info("开始模型训练")
        logger.info("="*50)
        
        # 设置模型和损失函数
        model, train_loss = self.setup_model_and_loss()
        
        # 创建数据加载器
        batch_size = self.config['batch_size']
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        logger.info(f"训练参数: 批次大小={batch_size}, "
                   f"训练轮数={self.config['num_epochs']}, "
                   f"学习率={self.config['learning_rate']}")
        
        # 创建评估器
        evaluator = self.create_evaluator(eval_examples)
        
        # 设置输出路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"models/finetuned_bge_{timestamp}"
        
        # 计算warmup步数
        total_steps = len(train_dataloader) * self.config['num_epochs']
        warmup_steps = int(total_steps * 0.1)
        
        logger.info(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
        logger.info(f"模型将保存到: {output_path}")
        
        # 开始训练
        logger.info("开始训练...")
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.config['num_epochs'],
            evaluation_steps=500,  # 每500步评估一次
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            optimizer_params={'lr': self.config['learning_rate']},
            show_progress_bar=True,
            use_amp=False  # 禁用自动混合精度，避免潜在问题
        )
        
        logger.info("模型训练完成!")
        logger.info(f"最佳模型保存在: {output_path}")
        
        # 保存训练配置
        self.save_training_config(output_path, train_examples, eval_examples)
        
        return output_path
    
    def save_training_config(self, model_path: str, 
                           train_examples: List[InputExample],
                           eval_examples: List[InputExample]):
        """保存训练配置和统计信息"""
        
        config_info = {
            'model_path': model_path,
            'base_model': self.config['base_model'],
            'training_config': self.config,
            'training_stats': {
                'train_samples': len(train_examples),
                'eval_samples': len(eval_examples),
                'train_positive_ratio': sum(1 for ex in train_examples if ex.label > 0.5) / len(train_examples),
                'eval_positive_ratio': sum(1 for ex in eval_examples if ex.label > 0.5) / len(eval_examples)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = os.path.join(model_path, 'training_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练配置已保存到: {config_file}")
        
        # 同时保存到outputs目录
        backup_file = f"outputs/models/training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        return config_file
    
    def run_training_pipeline(self) -> str:
        """运行完整的训练流程"""
        try:
            # 1. 加载预处理数据
            train_examples, eval_examples, test_examples = self.load_processed_data()
            
            # 2. 训练模型
            model_path = self.train_model(train_examples, eval_examples)
            
            logger.info("="*50)
            logger.info("训练流程完成!")
            logger.info("="*50)
            
            return model_path
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            raise

def main():
    """主函数 - 单独运行模型训练"""
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        logger.info(f"使用GPU训练: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("未检测到CUDA，将使用CPU训练（速度较慢）")
    
    # 检查预处理数据
    data_file = 'outputs/data/enhanced_processed_datasets.json'
    if not os.path.exists(data_file):
        logger.error(f"未找到预处理数据: {data_file}")
        logger.info("请先运行: python data_preparation.py")
        return
    
    # 训练配置
    training_config = {
        'base_model': 'models/bge-large-zh-v1.5',
        'batch_size': 8,  # 可根据GPU内存调整
        'num_epochs': 3,  # 增加到3轮
        'learning_rate': 2e-5,
        'max_grad_norm': 1.0
    }
    
    logger.info("训练配置:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")
    
    # 开始训练
    trainer = BGEModelTrainer(training_config)
    model_path = trainer.run_training_pipeline()
    
    print(f"\n训练完成!")
    print(f"模型保存位置: {model_path}")
    print(f"下一步运行: python model_evaluation.py")

if __name__ == "__main__":
    main()