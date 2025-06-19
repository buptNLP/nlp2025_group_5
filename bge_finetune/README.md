# BGE模型微调模块

本模块负责对BGE-large-zh-v1.5模型进行领域专业化微调，是RAG系统构建的**核心技术环节**。

## 项目关系

此模块承接数据爬虫模块的输出，为RAG系统的检索服务提供领域优化的embedding模型：

```
爬虫数据 → BGE模型微调 → RAG系统检索优化
```

主要完成：

- 构建正负样本训练集
- 创新性困难负样本挖掘
- LoRA方法微调BGE模型
- 模型性能评估与对比

## 文件说明

### 核心模块

```
├── enhanced_data_preparation.py    # 增强数据准备主模块
├── hard_negative_mining.py         # 困难负样本挖掘器
├── model_training.py               # 模型训练模块  
├── model_evaluation.py             # 模型评估模块
```

### 数据结构

```
├── data/                           # 输入数据
│   ├── csdn_articles_filtered.json # 爬虫输出的文章元信息
│   ├── filtered_qa_pairs.json      # 爬虫生成的QA对
│   └── article_texts/              # 文章正文目录
├── models/                         # 模型存储
│   ├── bge-large-zh-v1.5/          # 原始BGE模型
│   └── finetuned_bge_*/            # 微调后的模型
└── outputs/                        # 输出结果
    ├── data/                       # 处理后的训练数据
    ├── analysis/                   # 样本分析结果
    └── evaluation/                 # 模型评估报告
```

## 环境依赖

```bash
pip install sentence-transformers torch sklearn matplotlib seaborn jieba
```

需要NVIDIA GPU (推荐24GB显存)，支持CUDA 11.6+。

## 使用方法

### 1. 数据准备与困难负样本挖掘

```bash
python enhanced_data_preparation.py
```

**功能特色**：

- 构建正样本对（QA对+文档结构化样本）
- 随机采样生成基础负样本（70%为困难负样本）
- **五种策略困难负样本挖掘**：语义相似度、关键词重叠、TF-IDF、跨子领域、长度相似
- 智能文本截断保持段落完整性

### 2. 模型微调训练

```bash
python model_training.py
```

**技术亮点**：

- 基于LoRA方法的参数高效微调
- CosineSimilarityLoss损失函数
- 信息检索评估器实时监控
- 自动保存最佳模型检查点

### 3. 性能评估对比

```bash
python model_evaluation.py
```

**评估指标**：

- 准确率、相似度分离度、AUC等核心指标
- 原始vs微调模型全面对比
- 可视化分布图表生成
- 详细性能分析报告

## 参数配置

### 困难负样本配置

```python
hard_negative_config = {
    'enabled': True,
    'strategies': ['semantic', 'keyword', 'tfidf', 'cross_domain'],
    'ratio_to_positive': 0.8,
    'random_negative_ratio': 0.3
}
```

### 数据路径配置

```python
data_config = {
    'articles_file': 'data/csdn_articles_filtered.json',
    'qa_pairs_file': 'data/filtered_qa_pairs.json', 
    'article_texts_dir': 'data/article_texts/'
}
```

## 注意事项

1. **硬件要求**：推荐24GB GPU显存，16GB内存
2. **数据依赖**：需要先运行爬虫模块获取训练数据
3. **模型下载**：首次运行会自动下载BGE-large-zh-v1.5模型
4. **训练时间**：完整训练约需2-4小时（取决于硬件配置）

## 输出文件

- `outputs/data/enhanced_processed_datasets.json` - 训练数据集
- `models/finetuned_bge_*/` - 微调后的模型文件
- `outputs/evaluation/evaluation_results_*.json` - 性能评估报告
- `outputs/analysis/hard_negatives_analysis.json` - 困难负样本分析