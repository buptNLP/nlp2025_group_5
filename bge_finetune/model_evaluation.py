# model_evaluation.py - 模型评估模块
import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.setup_directories()
        
    def setup_directories(self):
        """创建评估相关目录"""
        directories = ['outputs/evaluation', 'outputs/evaluation/plots']
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
    
    def find_latest_model(self) -> str:
        """查找最新的微调模型"""
        models_dir = Path('models')
        if not models_dir.exists():
            raise FileNotFoundError("未找到models目录")
        
        # 查找所有微调模型目录
        model_dirs = [d for d in models_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('finetuned_bge_')]
        
        if not model_dirs:
            raise FileNotFoundError("未找到微调模型")
        
        # 按时间戳排序，取最新的
        latest_model = max(model_dirs, key=lambda x: x.name.split('_')[-2:])
        
        logger.info(f"找到最新模型: {latest_model}")
        return str(latest_model)
    
    def load_models(self, finetuned_model_path: str = None) -> Tuple[SentenceTransformer, SentenceTransformer]:
        """加载原始模型和微调模型"""
        logger.info("加载模型...")
        
        # 加载原始模型
        original_model = SentenceTransformer('models/bge-large-zh-v1.5')
        logger.info("原始模型加载完成")
        
        # 加载微调模型
        if finetuned_model_path is None:
            finetuned_model_path = self.find_latest_model()
        
        finetuned_model = SentenceTransformer(finetuned_model_path)
        logger.info(f"微调模型加载完成: {finetuned_model_path}")
        
        return original_model, finetuned_model
    
    def compute_similarities(self, model: SentenceTransformer, 
                           test_examples: List[InputExample]) -> Tuple[np.ndarray, List[float]]:
        """计算相似度分数"""
        logger.info("计算模型相似度...")
        
        # 提取文本和标签
        queries = [ex.texts[0] for ex in test_examples]
        documents = [ex.texts[1] for ex in test_examples]
        labels = [ex.label for ex in test_examples]
        
        # 编码文本
        query_embeddings = model.encode(queries, show_progress_bar=True, batch_size=32)
        doc_embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
        
        # 计算对角线相似度（每个query与对应doc的相似度）
        similarities = np.array([
            cosine_similarity([q_emb], [d_emb])[0, 0] 
            for q_emb, d_emb in zip(query_embeddings, doc_embeddings)
        ])
        
        return similarities, labels
    
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
        
        # 5. AUC近似计算（排序准确率）
        # 对于每个正样本，计算有多少负样本的相似度低于它
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
        
        # 尝试不同的阈值
        thresholds = np.linspace(0.1, 0.9, 81)  # 从0.1到0.9，步长0.01
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
                      finetuned_metrics: Dict[str, float]) -> Dict[str, Dict]:
        """对比两个模型的性能"""
        
        comparison = {}
        
        for metric_name in original_metrics:
            if metric_name in finetuned_metrics:
                original_value = original_metrics[metric_name]
                finetuned_value = finetuned_metrics[metric_name]
                
                # 计算绝对和相对改进
                absolute_improvement = finetuned_value - original_value
                
                if original_value != 0:
                    relative_improvement = (absolute_improvement / abs(original_value)) * 100
                else:
                    relative_improvement = 0.0
                
                comparison[metric_name] = {
                    'original': float(original_value),
                    'finetuned': float(finetuned_value),
                    'absolute_improvement': float(absolute_improvement),
                    'relative_improvement': float(relative_improvement)
                }
        
        return comparison
    
    def print_comparison_results(self, comparison: Dict[str, Dict]):
        """打印对比结果"""
        
        print("\n" + "="*80)
        print("模型性能对比结果")
        print("="*80)
        
        print(f"{'指标':<25} {'原始模型':<12} {'微调模型':<12} {'绝对提升':<12} {'相对提升':<12}")
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
                print(f"{metric:<25} {data['original']:<12.4f} {data['finetuned']:<12.4f} "
                      f"{data['absolute_improvement']:<12.4f} {data['relative_improvement']:<12.2f}%")
        
        # 再显示其他指标
        other_metrics = [m for m in comparison if m not in priority_metrics]
        if other_metrics:
            print("-"*80)
            for metric in other_metrics:
                data = comparison[metric]
                print(f"{metric:<25} {data['original']:<12.4f} {data['finetuned']:<12.4f} "
                      f"{data['absolute_improvement']:<12.4f} {data['relative_improvement']:<12.2f}%")
        
        print("="*80)
        
        # 总结关键改进
        print("\n关键改进总结:")
        
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
    
    def create_visualization(self, original_sims: np.ndarray, original_labels: List[float],
                           finetuned_sims: np.ndarray, finetuned_labels: List[float]):
        """创建可视化图表"""
        
        logger.info("创建可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 相似度分布对比
        ax1 = axes[0, 0]
        
        # 原始模型
        pos_mask_orig = np.array(original_labels) > 0.5
        neg_mask_orig = np.array(original_labels) <= 0.5
        
        ax1.hist(original_sims[pos_mask_orig], bins=30, alpha=0.7, label='Original Positive', color='blue')
        ax1.hist(original_sims[neg_mask_orig], bins=30, alpha=0.7, label='Original Negative', color='red')
        
        ax1.set_title('Original Model - Similarity Distribution')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 微调模型相似度分布
        ax2 = axes[0, 1]
        
        pos_mask_fine = np.array(finetuned_labels) > 0.5
        neg_mask_fine = np.array(finetuned_labels) <= 0.5
        
        ax2.hist(finetuned_sims[pos_mask_fine], bins=30, alpha=0.7, label='Finetuned Positive', color='green')
        ax2.hist(finetuned_sims[neg_mask_fine], bins=30, alpha=0.7, label='Finetuned Negative', color='orange')
        
        ax2.set_title('Finetuned Model - Similarity Distribution')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 箱线图对比
        ax3 = axes[1, 0]
        
        data_to_plot = [
            original_sims[pos_mask_orig], original_sims[neg_mask_orig],
            finetuned_sims[pos_mask_fine], finetuned_sims[neg_mask_fine]
        ]
        
        labels = ['Orig_Pos', 'Orig_Neg', 'Fine_Pos', 'Fine_Neg']
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'moccasin']
        
        box_plot = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('Similarity Distribution Comparison')
        ax3.set_ylabel('Cosine Similarity')
        ax3.grid(True, alpha=0.3)
        
        # 4. 散点图对比
        ax4 = axes[1, 1]
        
        # 创建比较数据（假设测试集相同）
        ax4.scatter(original_sims[pos_mask_orig], finetuned_sims[pos_mask_orig], 
                   alpha=0.6, label='Positive Samples', color='green')
        ax4.scatter(original_sims[neg_mask_orig], finetuned_sims[neg_mask_orig], 
                   alpha=0.6, label='Negative Samples', color='red')
        
        # 添加对角线
        min_sim = min(np.min(original_sims), np.min(finetuned_sims))
        max_sim = max(np.max(original_sims), np.max(finetuned_sims))
        ax4.plot([min_sim, max_sim], [min_sim, max_sim], 'k--', alpha=0.5, label='y=x')
        
        ax4.set_xlabel('Original Model Similarity')
        ax4.set_ylabel('Finetuned Model Similarity')
        ax4.set_title('Similarity Score Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f"outputs/evaluation/plots/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"可视化图表已保存: {plot_file}")
        
        plt.show()
        
        return plot_file
    
    def save_evaluation_results(self, comparison: Dict[str, Dict], 
                              finetuned_model_path: str) -> str:
        """保存评估结果"""
        
        # 准备保存的数据（确保所有值都是JSON可序列化的）
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'finetuned_model_path': finetuned_model_path,
            'original_model': 'models/bge-large-zh-v1.5',
            'metrics_comparison': comparison,
            'summary': {
                'total_metrics': len(comparison),
                'improved_metrics': sum(1 for m in comparison.values() if m['relative_improvement'] > 0),
                'declined_metrics': sum(1 for m in comparison.values() if m['relative_improvement'] < 0)
            }
        }
        
        # 保存结果
        results_file = f"outputs/evaluation/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存: {results_file}")
        
        return results_file
    
    def run_evaluation(self, finetuned_model_path: str = None) -> Dict:
        """运行完整的评估流程"""
        
        logger.info("="*50)
        logger.info("开始模型评估")
        logger.info("="*50)
        
        try:
            # 1. 加载测试数据
            test_examples = self.load_test_data()
            
            # 2. 加载模型
            original_model, finetuned_model = self.load_models(finetuned_model_path)
            
            # 3. 计算相似度
            logger.info("评估原始模型...")
            original_sims, original_labels = self.compute_similarities(original_model, test_examples)
            
            logger.info("评估微调模型...")
            finetuned_sims, finetuned_labels = self.compute_similarities(finetuned_model, test_examples)
            
            # 4. 计算指标
            logger.info("计算评估指标...")
            original_metrics = self.calculate_metrics(original_sims, original_labels)
            finetuned_metrics = self.calculate_metrics(finetuned_sims, finetuned_labels)
            
            # 5. 对比结果
            comparison = self.compare_models(original_metrics, finetuned_metrics)
            
            # 6. 显示结果
            self.print_comparison_results(comparison)
            
            # 7. 创建可视化
            plot_file = self.create_visualization(
                original_sims, original_labels,
                finetuned_sims, finetuned_labels
            )
            
            # 8. 保存结果
            results_file = self.save_evaluation_results(
                comparison, 
                finetuned_model_path if finetuned_model_path else self.find_latest_model()
            )
            
            logger.info("="*50)
            logger.info("评估完成!")
            logger.info("="*50)
            
            return {
                'comparison': comparison,
                'results_file': results_file,
                'plot_file': plot_file
            }
            
        except Exception as e:
            logger.error(f"评估过程失败: {e}")
            raise

def main():
    """主函数 - 单独运行模型评估"""
    
    # 检查是否存在测试数据
    data_file = 'outputs/data/enhanced_processed_datasets.json'
    if not os.path.exists(data_file):
        logger.error(f"未找到测试数据: {data_file}")
        logger.info("请先运行: python data_preparation.py")
        return
    
    # 检查是否存在微调模型
    models_dir = Path('models')
    if not models_dir.exists():
        logger.error("未找到models目录")
        logger.info("请先运行: python model_training.py")
        return
    
    # 开始评估
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()
    
    print(f"\n评估完成!")
    print(f"详细结果已保存到: {results['results_file']}")
    print(f"可视化图表: {results['plot_file']}")

if __name__ == "__main__":
    main()