from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = str(BASE_DIR / "models")
ARTICLES_DIR = BASE_DIR / "articles"
DATA_FILE = BASE_DIR / "data" / "csdn_articles_filtered.json"
UPLOAD_DIR = BASE_DIR / "uploads"

# 确保上传目录存在
UPLOAD_DIR.mkdir(exist_ok=True)

# 模型配置
BGE_MODEL_PATH = str(BASE_DIR / "models" / "finetuned_bge_20250616_211201")
BGE_RERANKER_PATH = str(BASE_DIR / "models" / "bge-reranker-v2-m3")

# API配置
API_BASE = ""
API_KEY = ""  # 请替换为你的API key
MODEL = ""  # 请替换为实际模型

# 文本分块配置（默认值）
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# 检索配置（默认值）
DEFAULT_TOP_K = 10
DEFAULT_RERANK_TOP_K = 3

# 角色设定
ML_ASSISTANT_ROLE = """你是一个专业的机器学习助手。你只回答与机器学习、深度学习、数据科学相关的问题。

规则：
1. 如果用户问题与机器学习无关，请礼貌地告知："我是专门的机器学习助手，只能回答机器学习、深度学习、数据科学相关的问题。请问您有什么机器学习方面的问题吗？"
2. 对于机器学习相关问题，请基于提供的参考文档给出专业、准确的回答
3. 回答要清晰、有条理，包含必要的技术细节
4. 如果参考文档不足以回答问题，请明确指出"""

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的机器学习助手。请基于提供的参考文档回答用户问题。要求：
1. 必须严格基于参考文档内容来回答
2. 回答要分成两部分：
   - 首先总结参考文档的相关内容
   - 然后基于这些内容组织完整的回答
3. 回答中要引用原文的关键句子，并标注出处
"""
