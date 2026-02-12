import os
from typing import List
import logging  # <--- 新增

# 导入 LangChain 相关组件
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from config import DOUBAO_API_KEY, DOUBAO_BASE_URL, MODEL_NAME, VECTOR_SEARCH_K
from prompts import get_rag_prompt
# 引入具体的错误类型 (这是为了捕获特定的 API 错误)
from openai import APIConnectionError, RateLimitError


# 获取 logger 实例
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 第一把刀：数据加载与切分逻辑
# ---------------------------------------------------------
def load_and_split_document(file_path: str, chunk_size: int = 512, chunk_overlap: int = 200) -> List[Document]:
    """加载PDF文档并进行文本切分"""
    # 替换 print 为 logger.info
    logger.info(f"正在加载文档: {file_path}")
    loader = PyMuPDFLoader(file_path)
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    # 按照你原代码的逻辑，这里仅做前1页的切分演示
    texts = text_splitter.create_documents([page.page_content for page in pages[:1]])
    logger.info(f"文档切分完成，共 {len(texts)} 个切片")
    return texts

# ---------------------------------------------------------
# 第二把刀：向量数据库构建逻辑
# ---------------------------------------------------------
def build_vector_store(documents: List[Document]) -> FAISS:
    """将文档片段向量化并存入 FAISS 向量数据库"""
    print("🧠 正在构建向量数据库...")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 安全读取你的 API Key
    )
    db = FAISS.from_documents(documents, embeddings)
    logger.info("✅ 向量数据库构建完成。")
    return db

# ---------------------------------------------------------
# 第三把刀：核心 RAG 生成逻辑 (LCEL)
# ---------------------------------------------------------
def generate_rag_response(query: str, vector_store: FAISS) -> str:
    """核心 RAG 生成逻辑"""
    logger.info("正在调用大模型 API...")
    retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})
    
    # 直接使用 config 里的常量
    llm = ChatOpenAI(
        temperature=0.5,
        base_url=DOUBAO_BASE_URL,
        api_key=DOUBAO_API_KEY,
        model=MODEL_NAME,
    )
    
    # 直接从 prompts 模块获取提示词模板
    prompt = get_rag_prompt()
    
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 🌟 关键防御：给最容易出错的 API 调用加上“防弹衣”
    try:
        return rag_chain.invoke(query)
        
    except APIConnectionError:
        logger.error("❌ 网络连接失败，请检查网线或代理设置")
        return "抱歉，我连不上大脑了（网络错误）。"
        
    except RateLimitError:
        logger.error("❌ API 调用太快被限流了")
        return "我有点累，请稍后再试（限流）。"
        
    except Exception as e:
        # 捕获所有其他未知的错误
        logger.error(f"❌ 未知错误: {str(e)}")
        raise e  # 抛出异常，让 main.py 里的总开关去处理

