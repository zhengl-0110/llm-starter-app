import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 

# 关键修复1：调用 load_dotenv() 加载 .env 文件中的环境变量
load_dotenv()

# 获取 API Key 并校验
api_key = os.getenv("DOUBAO_API_KEY", "")
if not api_key:
    raise ValueError("未配置 DOUBAO_API_KEY 环境变量，请检查 .env 文件或系统环境变量")

# 初始化大模型
llm = ChatOpenAI(
    temperature=0.5,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1-6-251015",
)

# 调用模型并获取响应
response = llm.invoke("你好")
print(response.content)