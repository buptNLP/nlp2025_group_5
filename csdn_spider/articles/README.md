# 机器学习知识库数据爬虫模块

本模块负责从CSDN平台爬取机器学习相关技术博客，为RAG系统构建知识库提供高质量数据源。

## 项目关系

此爬虫模块是"基于LangChain搭建的机器学习本地知识库RAG问答系统"课程设计项目的**数据采集阶段**，主要完成：

- 从CSDN机器学习板块爬取文章列表和元信息
- 获取每篇文章的完整正文内容
- 自动生成问答对用于模型微调
- 为后续BGE模型微调和RAG系统提供训练数据

## 文件说明

```
├── csdn_spider.py              # 文章列表爬虫
├── single_article.py           # 单篇文章内容爬虫
├── qa.py                       # QA对自动生成器
├── csdn_articles_filtered.json # 筛选后的文章元信息
├── filtered_qa_pairs.json      # 筛选后的问答对
├── generated_qa_pairs.json     # 原始问答对
└── articles/                   # 文章正文内容目录
    ├── 138348212.txt
    ├── 137581920.txt
    └── ...
```

## 环境依赖

```bash
pip install selenium undetected-chromedriver beautifulsoup4 requests
```

需要安装Chrome浏览器。

## 使用方法

### 1. 爬取文章列表

```bash
python csdn_spider.py
```

- 自动滚动加载更多文章
- 增量爬取，避免重复
- 输出：`csdn_articles.json`

### 2. 爬取文章内容

```bash
python single_article.py
```

- 基于文章列表获取完整正文
- 智能处理"阅读全文"限制
- 输出：`articles/` 目录下的txt文件

### 3. 生成问答对

```bash
# 修改qa.py中的API_KEY
python qa.py
```

- 基于文章内容调用API生成QA对
- 自动过滤和质量控制
- 输出：`generated_qa_pairs.json` 和 `filtered_qa_pairs.json`

## 配置说明

### 爬虫参数

- `max_scrolls`: 页面滚动次数，控制爬取数量
- `headless`: 是否无头模式运行

### QA生成参数

- `API_KEY`: ChatGLM API密钥
- `max_workers`: 并发线程数
- `max_articles`: 处理文章数量限制

## 数据输出

最终产出用于RAG系统的数据：

- **高质量文章**：经过筛选的机器学习技术博客
- **QA对**：用于BGE模型微调的训练数据
- **元信息**：标题、链接、阅读量等文章统计信息

## 注意事项

1. 遵守CSDN的robots.txt规则，控制爬取频率
2. API调用需要稳定的网络环境
3. 建议分批次运行，避免单次爬取过多数据
4. 爬虫可能需要根据网站结构变化进行调整