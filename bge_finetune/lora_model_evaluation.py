# lora_model_evaluation.py - LoRA模型评估
import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from sentence_transformers import InputExample
import matplotlib.pyplot as plt
import seaborn as sns

# 导入LoRA训练模块中的组件
from lora_model_training import LoRABGEModel, SimilarityDataset, load_lora_model_for_inference

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRAModelEvaluator:
    """LoRA模型评估器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        
    def setup_directories(self):
        """创建评估相关目录"""
        directories = ['outputs/lora_evaluation', 'outputs/lora_evaluation/plots']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self) -> List[InputExample]:
        """加载测试数据"""
        logger.info("加载测试数据...")
        
        data_file = 'outputs/data/enhanced_processed_datasets.json'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"未找到预处理数据文件: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        
        # 转换测试数据为InputExample格式
        test_examples = []
        for item in datasets['test']:
            example = InputExample(
                texts=[item['text1'], item['text2']],
                label=item['label']
            )
            test_examples.append(example)
        
        logger.info(f"测试数据加载完成: {len(test_examples)} 个样本")
        return test_examples
    
    def find_latest_lora_model(self) -> str:
        """查找最新的LoRA微调模型"""
        models_dir = Path('models')
        if not models_dir.exists():
            raise FileNotFoundError("未找到models目录")
        
        # 查找所有LoRA微调模型目录
        model_dirs = [d for d in models_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('lora_finetuned_bge_')]
        
        if not model_dirs:
            raise FileNotFoundError("未找到LoRA微调模型")
        
        # 按时间戳排序，取最新的
        latest_model = max(model_dirs, key=lambda x: x.name.split('_')[-2:])
        
        logger.info(f"找到最新LoRA模型: {latest_model}")
        return str(latest_model)
    
    def load_original_model(self, base_model_path: str):
        """加载原始BGE模型"""
        logger.info(f"加载原始模型: {base_model_path}")
        
        model = AutoModel.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        model.to(self.device)
        model.eval()
        
        logger.info("原始模型加载完成")
        return model, tokenizer
    
    def load_lora_model(self, lora_model_path: str):
        """加载LoRA微调模型"""
        logger.info(f"加载LoRA模型: {lora_model_path}")
        
        model, tokenizer = load_lora_model_for_inference(lora_model_path)
        model.to(self.device)
        model.eval()
        
        logger.info("LoRA模型加载完成")
        return model, tokenizer
    
    def mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def compute_similarities_original(self, model, tokenizer, test_examples: List[InputExample]) -> Tuple[np.ndarray, List[float]]:
        """计算原始模型的相似度分数"""
        logger.info("计算原始模型相似度...")
        
        similarities = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for example in test_examples:
                text1, text2 = example.texts[0], example.texts[1]
                label = example.label
                
                # 编码文本
                encoding1 = tokenizer(text1, truncation=True, padding=True, 
                                    max_length=512, return_tensors='pt').to(self.device)
                encoding2 = tokenizer(text2, truncation=True, padding=True, 
                                    max_length=512, return_tensors='pt').to(self.device)
                
                # 获取embeddings
                output1 = model(**encoding1)
                output2 = model(**encoding2)
                
                emb1 = self.mean_pooling(output1, encoding1['attention_mask'])
                emb2 = self.mean_pooling(output2, encoding2['attention_mask'])
                
                # L2归一化
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(emb1, emb2).item()
                
                similarities.append(similarity)
                labels.append(label)
        
        return np.array(similarities), labels
    
    def compute_similarities_lora(self, model, tokenizer, test_examples: List[InputExample]) -> Tuple[np.ndarray, List[float]]:
        """计算LoRA模型的相似度分数"""
        logger.info("计算LoRA模型相似度...")
        
        similarities = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for example in test_examples:
                text1, text2 = example.texts[0], example.texts[1]
                label = example.label
                
                # 编码文本
                encoding1 = tokenizer(text1, truncation=True, padding=True, 
                                    max_length=512, return_tensors='pt').to(self.device)
                encoding2 = tokenizer(text2, truncation=True, padding=True, 
                                    max_length=512, return_tensors='pt').to(self.device)
                
                # 获取embeddings
                output1 = model(**encoding1)
                output2 = model(**encoding2)
                
                emb1 = self.mean_pooling(output1, encoding1['attention_mask'])
                emb2 = self.mean_pooling(output2, encoding2['attention_mask'])
                
                # L2归一化
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(emb1, emb2).item()
                
                similarities.append(similarity)
                labels.append(label)
        
        return np.array(similarities), labels
    
    def calculate_metrics(self, similarities: np.ndarray, labels: List[float]) -> Dict[str, float]:
        """计算评估指标"""
        
        # 转换为numpy数组
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # 基础统计
        positive_mask = labels > 0.5
        negative_mask = labels <= 0.5
        
        positive_sims = similarities[positive_mask]
        negative_sims = similarities[negative_mask]
        
        metrics = {}
        
        # 1. 平均相似度
        metrics['avg_positive_similarity'] = float(np.mean(positive_sims)) if len(positive_sims) > 0 else 0.0
        metrics['avg_negative_similarity'] = float(np.mean(negative_sims)) if len(negative_sims) > 0 else 0.0
        
        # 2. 相似度分离度
        if len(positive_sims) > 0 and len(negative_sims) > 0:
            metrics['similarity_separation'] = float(np.mean(positive_sims) - np.mean(negative_sims))
        else:
            metrics['similarity_separation'] = 0.0
        
        # 3. 准确率（使用0.5作为阈值）
        threshold = 0.5
        predictions = (similarities > threshold).astype(float)
        accuracy = float(np.mean(predictions == labels))
        metrics['accuracy'] = accuracy
        
        # 4. 最佳阈值下的准确率
        best_threshold, best_accuracy = self.find_best_threshold(similarities, labels)
        metrics['best_threshold'] = float(best_threshold)
        metrics['best_accuracy'] = float(best_accuracy)
        
        # 5. AUC近似计算
        auc_scores = []
        for pos_sim in positive_sims:
            auc_score = float(np.mean(negative_sims < pos_sim))
            auc_scores.append(auc_score)
        
        metrics['auc_approx'] = float(np.mean(auc_scores)) if auc_scores else 0.0
        
        # 6. 分位数统计
        if len(positive_sims) > 0:
            metrics['positive_median'] = float(np.median(positive_sims))
            metrics['positive_q75'] = float(np.percentile(positive_sims, 75))
            metrics['positive_q25'] = float(np.percentile(positive_sims, 25))
        
        if len(negative_sims) > 0:
            metrics['negative_median'] = float(np.median(negative_sims))
            metrics['negative_q75'] = float(np.percentile(negative_sims, 75))
            metrics['negative_q25'] = float(np.percentile(negative_sims, 25))
        
        return metrics
    
    def find_best_threshold(self, similarities: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """寻找最佳分类阈值"""
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_accuracy = 0.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (similarities > threshold).astype(float)
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy
    
    def compare_models(self, original_metrics: Dict[str, float], 
                      lora_metrics: Dict[str, float]) -> Dict[str, Dict]:
        """对比两个模型的性能"""
        
        comparison = {}
        
        for metric_name in original_metrics:
            if metric_name in lora_metrics:
                original_value = original_metrics[metric_name]
                lora_value = lora_metrics[metric_name]
                
                # 计算绝对和相对改进
                absolute_improvement = lora_value - original_value
                
                if original_value != 0:
                    relative_improvement = (absolute_improvement / abs(original_value)) * 100
                else:
                    relative_improvement = 0.0
                
                comparison[metric_name] = {
                    'original': float(original_value),
                    'lora_finetuned': float(lora_value),
                    'absolute_improvement': float(absolute_improvement),
                    'relative_improvement': float(relative_improvement)
                }
        
        return comparison
    
    def print_comparison_results(self, comparison: Dict[str, Dict]):
        """打印对比结果"""
        
        print("\n" + "="*80)
        print("LoRA模型性能对比结果")
        print("="*80)
        
        print(f"{'指标':<25} {'原始模型':<12} {'LoRA模型':<12} {'绝对提升':<12} {'相对提升':<12}")
        print("-"*80)
        
        # 重要指标优先显示
        priority_metrics = [
            'accuracy', 'best_accuracy', 'auc_approx', 'similarity_separation',
            'avg_positive_similarity', 'avg_negative_similarity'
        ]
        
        # 先显示重要指标
        for metric in priority_metrics:
            if metric in comparison:
                data = comparison[metric]
                print(f"{metric:<25} {data['original']:<12.4f} {data['lora_finetuned']:<12.4f} "
                      f"{data['absolute_improvement']:<12.4f} {data['relative_improvement']:<12.2f}%")
        
        # 再显示其他指标
        other_metrics = [m for m in comparison if m not in priority_metrics]
        if other_metrics:
            print("-"*80)
            for metric in other_metrics:
                data = comparison[metric]
                print(f"{metric:<25} {data['original']:<12.4f} {data['lora_finetuned']:<12.4f} "
                      f"{data['absolute_improvement']:<12.4f} {data['relative_improvement']:<12.2f}%")
        
        print("="*80)
        
        # 总结关键改进
        print("\nLoRA微调关键改进总结:")
        
        if 'accuracy' in comparison:
            acc_imp = comparison['accuracy']['relative_improvement']
            print(f"• 准确率提升: {acc_imp:+.2f}%")
        
        if 'similarity_separation' in comparison:
            sep_imp = comparison['similarity_separation']['absolute_improvement']
            print(f"• 相似度分离度提升: {sep_imp:+.4f}")
        
        if 'avg_negative_similarity' in comparison:
            neg_change = comparison['avg_negative_similarity']['relative_improvement']
            if neg_change < 0:
                print(f"• 负样本相似度降低: {abs(neg_change):.2f}%")
            else:
                print(f"• 负样本相似度变化: {neg_change:+.2f}%")
        
    
    def create_lora_visualization(self, original_sims: np.ndarray, original_labels: List[float],
                                 lora_sims: np.ndarray, lora_labels: List[float]):
        """创建LoRA对比可视化图表"""
        
        logger.info("创建LoRA对比可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 原始模型相似度分布
        ax1 = axes[0, 0]
        
        pos_mask_orig = np.array(original_labels) > 0.5
        neg_mask_orig = np.array(original_labels) <= 0.5
        
        ax1.hist(original_sims[pos_mask_orig], bins=30, alpha=0.7, label='Original Positive', color='blue')
        ax1.hist(original_sims[neg_mask_orig], bins=30, alpha=0.7, label='Original Negative', color='red')
        
        ax1.set_title('Original BGE Model - Similarity Distribution')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. LoRA模型相似度分布
        ax2 = axes[0, 1]
        
        pos_mask_lora = np.array(lora_labels) > 0.5
        neg_mask_lora = np.array(lora_labels) <= 0.5
        
        ax2.hist(lora_sims[pos_mask_lora], bins=30, alpha=0.7, label='LoRA Positive', color='green')
        ax2.hist(lora_sims[neg_mask_lora], bins=30, alpha=0.7, label='LoRA Negative', color='orange')
        
        ax2.set_title('LoRA Finetuned Model - Similarity Distribution')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 箱线图对比
        ax3 = axes[1, 0]
        
        data_to_plot = [
            original_sims[pos_mask_orig], original_sims[neg_mask_orig],
            lora_sims[pos_mask_lora], lora_sims[neg_mask_lora]
        ]
        
        labels = ['Orig_Pos', 'Orig_Neg', 'LoRA_Pos', 'LoRA_Neg']
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'moccasin']
        
        box_plot = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('LoRA vs Original - Similarity Distribution Comparison')
        ax3.set_ylabel('Cosine Similarity')
        ax3.grid(True, alpha=0.3)
        
        # 4. 散点图对比
        ax4 = axes[1, 1]
        
        ax4.scatter(original_sims[pos_mask_orig], lora_sims[pos_mask_orig], 
                   alpha=0.6, label='Positive Samples', color='green')
        ax4.scatter(original_sims[neg_mask_orig], lora_sims[neg_mask_lora], 
                   alpha=0.6, label='Negative Samples', color='red')
        
        # 添加对角线
        min_sim = min(np.min(original_sims), np.min(lora_sims))
        max_sim = max(np.max(original_sims), np.max(lora_sims))
        ax4.plot([min_sim, max_sim], [min_sim, max_sim], 'k--', alpha=0.5, label='y=x')
        
        ax4.set_xlabel('Original Model Similarity')
        ax4.set_ylabel('LoRA Model Similarity')
        ax4.set_title('Original vs LoRA Similarity Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f"outputs/lora_evaluation/plots/lora_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"LoRA对比图表已保存: {plot_file}")
        
        plt.show()
        
        return plot_file
    
    def save_lora_evaluation_results(self, comparison: Dict[str, Dict], 
                                    lora_model_path: str) -> str:
        """保存LoRA评估结果"""
        
        # 读取LoRA训练信息
        training_info_file = os.path.join(lora_model_path, 'training_info.json')
        lora_training_info = {}
        if os.path.exists(training_info_file):
            with open(training_info_file, 'r', encoding='utf-8') as f:
                lora_training_info = json.load(f)
        
        # 准备保存的数据
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'lora_model_path': lora_model_path,
            'original_model': lora_training_info.get('training_config', {}).get('base_model', 'models/bge-large-zh-v1.5'),
            'lora_config': lora_training_info.get('lora_config', {}),
            'training_config': lora_training_info.get('training_config', {}),
            'metrics_comparison': comparison,
            'lora_summary': {
                'total_metrics': len(comparison),
                'improved_metrics': sum(1 for m in comparison.values() if m['relative_improvement'] > 0),
                'declined_metrics': sum(1 for m in comparison.values() if m['relative_improvement'] < 0),
                'parameter_efficiency': {
                    'lora_rank': lora_training_info.get('lora_config', {}).get('r', 16),
                    'target_modules': lora_training_info.get('lora_config', {}).get('target_modules', []),
                    'trainable_ratio': "~0.1%"  # LoRA典型比例
                }
            }
        }
        
        # 保存结果
        results_file = f"outputs/lora_evaluation/lora_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"LoRA评估结果已保存: {results_file}")
        
        return results_file
    
    def run_lora_evaluation(self, lora_model_path: str = None) -> Dict:
        """运行完整的LoRA评估流程"""
        
        logger.info("="*60)
        logger.info("开始LoRA模型评估")
        logger.info("="*60)
        
        try:
            # 1. 加载测试数据
            test_examples = self.load_test_data()
            
            # 2. 确定LoRA模型路径
            if lora_model_path is None:
                lora_model_path = self.find_latest_lora_model()
            
            # 读取LoRA模型的基础模型信息
            training_info_file = os.path.join(lora_model_path, 'training_info.json')
            base_model_name = 'models/bge-large-zh-v1.5'  # 默认值
            
            if os.path.exists(training_info_file):
                with open(training_info_file, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                base_model_name = training_info.get('training_config', {}).get('base_model', base_model_name)
            
            # 3. 加载模型
            logger.info("加载原始模型...")
            original_model, original_tokenizer = self.load_original_model(base_model_name)
            
            logger.info("加载LoRA模型...")
            lora_model, lora_tokenizer = self.load_lora_model(lora_model_path)
            
            # 4. 计算相似度
            original_sims, original_labels = self.compute_similarities_original(
                original_model, original_tokenizer, test_examples
            )
            
            lora_sims, lora_labels = self.compute_similarities_lora(
                lora_model, lora_tokenizer, test_examples
            )
            
            # 5. 计算指标
            logger.info("计算评估指标...")
            original_metrics = self.calculate_metrics(original_sims, original_labels)
            lora_metrics = self.calculate_metrics(lora_sims, lora_labels)
            
            # 6. 对比结果
            comparison = self.compare_models(original_metrics, lora_metrics)
            
            # 7. 显示结果
            self.print_comparison_results(comparison)
            
            # 8. 创建可视化
            plot_file = self.create_lora_visualization(
                original_sims, original_labels,
                lora_sims, lora_labels
            )
            
            # 9. 保存结果
            results_file = self.save_lora_evaluation_results(comparison, lora_model_path)
            
            logger.info("="*60)
            logger.info("LoRA评估完成!")
            logger.info("="*60)
            
            return {
                'comparison': comparison,
                'results_file': results_file,
                'plot_file': plot_file,
                'lora_model_path': lora_model_path
            }
            
        except Exception as e:
            logger.error(f"LoRA评估过程失败: {e}")
            raise

def main():
    """主函数 - 单独运行LoRA模型评估"""
    
    # 检查是否存在测试数据
    data_file = 'outputs/data/enhanced_processed_datasets.json'
    if not os.path.exists(data_file):
        logger.error(f"未找到测试数据: {data_file}")
        logger.info("请先运行: python enhanced_data_preparation.py")
        return
    
    # 检查是否存在LoRA微调模型
    models_dir = Path('models')
    if not models_dir.exists():
        logger.error("未找到models目录")
        logger.info("请先运行: python lora_model_training.py")
        return
    
    # 开始LoRA评估
    evaluator = LoRAModelEvaluator()
    results = evaluator.run_lora_evaluation()
    
    print(f"\nLoRA评估完成!")
    print(f"详细结果已保存到: {results['results_file']}")
    print(f"可视化图表: {results['plot_file']}")
    print(f"LoRA模型路径: {results['lora_model_path']}")

if __name__ == "__main__":
    main()