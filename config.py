# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 统一管理配置项
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = "doubao-seed-1-6-251015"
VECTOR_SEARCH_K = 5  # 检索返回的文档数量