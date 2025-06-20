# lora_model_training.py - LoRA微调实现
import os
import json
import logging
import torch
import torch.nn as nn
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sentence_transformers import InputExample
import torch.nn.functional as F

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityDataset(Dataset):
    """相似度训练数据集"""
    
    def __init__(self, examples: List[InputExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text1, text2 = example.texts[0], example.texts[1]
        label = float(example.label)
        
        # 编码文本
        encoding1 = self.tokenizer(
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

class LoRABGEModel(nn.Module):
    """LoRA增强的BGE模型"""
    
    def __init__(self, model_name: str, lora_config: LoraConfig):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        
        # 加载预训练模型
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # 应用LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        # 添加池化层（平均池化）
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]  # 第一个元素是token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_trainable_parameters(self):
        """获取可训练参数统计"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_param

class LoRABGETrainer:
    """LoRA BGE训练器"""
    
    def __init__(self, training_config: Dict):
        self.config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        
        # LoRA配置
        self.lora_config = LoraConfig(
            r=self.config.get('lora_rank', 16),  # LoRA秩
            lora_alpha=self.config.get('lora_alpha', 32),  # LoRA缩放参数
            target_modules=self.config.get('target_modules', [
                "query", "key", "value", "dense"
            ]),  # 目标模块
            lora_dropout=self.config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        logger.info(f"LoRA配置: rank={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")
        logger.info(f"目标模块: {self.lora_config.target_modules}")
    
    def setup_directories(self):
        """创建训练相关目录"""
        directories = ['models', 'outputs/models', 'outputs/logs', 'outputs/lora_checkpoints']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_processed_data(self) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
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
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        logger.info("初始化LoRA模型和分词器...")
        
        model_name = self.config['base_model']
        logger.info(f"加载基础模型: {model_name}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 创建LoRA模型
        model = LoRABGEModel(model_name, self.lora_config)
        model.to(self.device)
        
        # 打印参数统计
        trainable_params, total_params = model.get_trainable_parameters()
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"总参数: {total_params:,}")
        logger.info(f"LoRA参数占比: {100 * trainable_params / total_params:.3f}%")
        
        return model, tokenizer
    
    def create_data_loaders(self, train_examples: List[InputExample], 
                           eval_examples: List[InputExample], 
                           tokenizer) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        
        max_length = self.config.get('max_length', 512)
        batch_size = self.config['batch_size']
        
        train_dataset = SimilarityDataset(train_examples, tokenizer, max_length)
        eval_dataset = SimilarityDataset(eval_examples, tokenizer, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"数据加载器创建完成: 训练批次={len(train_loader)}, 验证批次={len(eval_loader)}")
        
        return train_loader, eval_loader
    
    def cosine_similarity_loss(self, embeddings1, embeddings2, labels):
        """余弦相似度损失函数"""
        # 计算余弦相似度
        similarities = F.cosine_similarity(embeddings1, embeddings2)
        
        # 将标签转换为相似度目标值
        targets = labels  # 正样本标签为1，负样本标签为0
        
        # 使用MSE损失
        loss = F.mse_loss(similarities, targets)
        return loss, similarities
    
    def evaluate_model(self, model, eval_loader):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                # 移动数据到设备
                input_ids1 = batch['input_ids1'].to(self.device)
                attention_mask1 = batch['attention_mask1'].to(self.device)
                input_ids2 = batch['input_ids2'].to(self.device)
                attention_mask2 = batch['attention_mask2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                embeddings1 = model(input_ids1, attention_mask1)
                embeddings2 = model(input_ids2, attention_mask2)
                
                # 计算损失
                loss, similarities = self.cosine_similarity_loss(embeddings1, embeddings2, labels)
                
                total_loss += loss.item()
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(eval_loader)
        
        # 计算准确率（使用0.5作为阈值）
        predictions = np.array(all_similarities) > 0.5
        labels = np.array(all_labels) > 0.5
        accuracy = np.mean(predictions == labels)
        
        # 计算正负样本的平均相似度
        positive_mask = labels
        negative_mask = ~labels
        
        avg_positive_sim = np.mean(np.array(all_similarities)[positive_mask]) if np.any(positive_mask) else 0
        avg_negative_sim = np.mean(np.array(all_similarities)[negative_mask]) if np.any(negative_mask) else 0
        
        separation = avg_positive_sim - avg_negative_sim
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'avg_positive_similarity': avg_positive_sim,
            'avg_negative_similarity': avg_negative_sim,
            'similarity_separation': separation
        }
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids1 = batch['input_ids1'].to(self.device)
            attention_mask1 = batch['attention_mask1'].to(self.device)
            input_ids2 = batch['input_ids2'].to(self.device)
            attention_mask2 = batch['attention_mask2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            embeddings1 = model(input_ids1, attention_mask1)
            embeddings2 = model(input_ids2, attention_mask2)
            
            # 计算损失
            loss, similarities = self.cosine_similarity_loss(embeddings1, embeddings2, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('max_grad_norm', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(step+1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches
    
    def save_lora_model(self, model, output_path: str, epoch: int, metrics: Dict):
        """保存LoRA模型"""
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        model.model.save_pretrained(output_path)
        
        # 保存训练信息
        training_info = {
            'epoch': epoch,
            'lora_config': self.lora_config.__dict__,
            'training_config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        info_file = os.path.join(output_path, 'training_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"LoRA模型已保存到: {output_path}")
    
    def train_model(self, train_examples: List[InputExample], 
                   eval_examples: List[InputExample]) -> str:
        """训练模型"""
        logger.info("="*60)
        logger.info("开始LoRA模型训练")
        logger.info("="*60)
        
        # 设置模型和分词器
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # 创建数据加载器
        train_loader, eval_loader = self.create_data_loaders(
            train_examples, eval_examples, tokenizer
        )
        
        # 设置优化器和调度器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        total_steps = len(train_loader) * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        logger.info(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
        
        # 设置输出路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"models/lora_finetuned_bge_{timestamp}"
        
        # 训练循环
        best_accuracy = 0
        best_epoch = -1
        training_history = []
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\n开始第 {epoch+1}/{self.config['num_epochs']} 轮训练")
            
            # 训练一个epoch
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, epoch)
            
            # 验证
            logger.info("验证模型...")
            eval_metrics = self.evaluate_model(model, eval_loader)
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **eval_metrics
            }
            training_history.append(epoch_info)
            
            # 打印结果
            logger.info(f"Epoch {epoch+1} 结果:")
            logger.info(f"  训练损失: {train_loss:.4f}")
            logger.info(f"  验证损失: {eval_metrics['loss']:.4f}")
            logger.info(f"  验证准确率: {eval_metrics['accuracy']:.4f}")
            logger.info(f"  相似度分离度: {eval_metrics['similarity_separation']:.4f}")
            logger.info(f"  正样本平均相似度: {eval_metrics['avg_positive_similarity']:.4f}")
            logger.info(f"  负样本平均相似度: {eval_metrics['avg_negative_similarity']:.4f}")
            
            # 保存最佳模型
            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                best_epoch = epoch
                self.save_lora_model(model, output_path, epoch, eval_metrics)
                logger.info(f"新的最佳模型! 准确率: {best_accuracy:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.config.get('save_every', 1) == 0:
                checkpoint_path = f"outputs/lora_checkpoints/checkpoint_epoch_{epoch+1}"
                self.save_lora_model(model, checkpoint_path, epoch, eval_metrics)
        
        # 保存训练历史
        history_file = os.path.join(output_path, 'training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        logger.info("="*60)
        logger.info("LoRA训练完成!")
        logger.info(f"最佳模型保存在: {output_path}")
        logger.info(f"最佳准确率: {best_accuracy:.4f} (第{best_epoch+1}轮)")
        logger.info("="*60)
        
        return output_path
    
    def run_lora_training_pipeline(self) -> str:
        """运行完整的LoRA训练流程"""
        try:
            # 1. 加载预处理数据
            train_examples, eval_examples, test_examples = self.load_processed_data()
            
            # 2. 训练LoRA模型
            model_path = self.train_model(train_examples, eval_examples)
            
            return model_path
            
        except Exception as e:
            logger.error(f"LoRA训练流程失败: {e}")
            raise

def load_lora_model_for_inference(lora_model_path: str, base_model_name: str = None):
    """加载LoRA模型用于推理"""
    
    if base_model_name is None:
        # 从训练信息中读取基础模型名称
        info_file = os.path.join(lora_model_path, 'training_info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            base_model_name = info['training_config']['base_model']
        else:
            base_model_name = 'models/bge-large-zh-v1.5'
    
    logger.info(f"加载LoRA模型: {lora_model_path}")
    logger.info(f"基础模型: {base_model_name}")
    
    # 加载基础模型
    base_model = AutoModel.from_pretrained(base_model_name)
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    return model, tokenizer

def main():
    """主函数 - 运行LoRA训练"""
    
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
        logger.info("请先运行: python enhanced_data_preparation.py")
        return
    
    # LoRA训练配置
    training_config = {
        'base_model': 'models/bge-large-zh-v1.5',
        'batch_size': 8,  # 可根据GPU内存调整
        'num_epochs': 3,
        'learning_rate': 5e-4,  # LoRA通常使用更高的学习率
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'warmup_ratio': 0.1,
        'max_length': 512,
        'save_every': 1,
        
        # LoRA特定配置
        'lora_rank': 16,        # LoRA秩
        'lora_alpha': 32,       # LoRA缩放参数
        'lora_dropout': 0.1,    # LoRA dropout
        'target_modules': [     # 目标模块
            "query", "key", "value", "dense"
        ]
    }
    
    logger.info("LoRA训练配置:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")
    
    # 开始LoRA训练
    trainer = LoRABGETrainer(training_config)
    model_path = trainer.run_lora_training_pipeline()
    
    print(f"\nLoRA训练完成!")
    print(f"模型保存位置: {model_path}")
    print(f"下一步可以运行: python lora_model_evaluation.py")

if __name__ == "__main__":
    main()