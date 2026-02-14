# core.py (Day 7 终极修正版)
import logging
from typing import List, Generator

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import APIConnectionError, RateLimitError 

# 确保 config.py 和 prompts.py 存在
from config import DOUBAO_API_KEY, DOUBAO_BASE_URL, MODEL_NAME, VECTOR_SEARCH_K
from prompts import get_rag_prompt

logger = logging.getLogger(__name__)

# --- 1. 加载函数 (保留了 chunk_size=1000 的优化) ---
def load_and_split_document(file_path: str) -> List[Document]:
    logger.info(f"正在加载文档: {file_path}")
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    loader = PyMuPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # 🌟 关键点：这里保留了之前调试出的最佳参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300, 
        length_function=len, 
        add_start_index=True,
    )
    texts = text_splitter.create_documents([page.page_content for page in pages])
    logger.info(f"文档切分完成，共 {len(texts)} 个片段")
    return texts

# --- 2. 建库函数 ---
def build_vector_store(documents: List[Document]) -> FAISS:
    logger.info("正在构建向量数据库...")
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    import os
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    db = FAISS.from_documents(documents, embeddings)
    logger.info("向量数据库构建完毕")
    return db

# --- 3. 内部流水线组装 (这里修复了报错的核心) ---
def _get_rag_chain(vector_store: FAISS):
    """
    组装完整的 RAG 流水线：
    Input(Query) -> Retriever(Context) + Query -> Prompt -> LLM -> Output
    """
    # 1. 检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})
    
    # 2. 模型
    llm = ChatOpenAI(
        temperature=0.1,
        base_url=DOUBAO_BASE_URL,
        api_key=DOUBAO_API_KEY,
        model=MODEL_NAME,
    )
    
    # 3. 提示词
    prompt = get_rag_prompt()
    
    # 4. 🔗 链条组装 (LCEL)
    # ❌ 之前的报错是因为这里漏掉了最前面的字典映射
    rag_chain = (
        # 下面这个字典就是把 user query 变成 prompt 需要的 context 和 question
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 4. 同步生成 (CLI 用) ---
def generate_rag_response(query: str, vector_store: FAISS) -> str:
    logger.info("正在调用 API (同步)...")
    try:
        chain = _get_rag_chain(vector_store)
        return chain.invoke(query)
    except Exception as e:
        logger.error(f"调用失败: {e}")
        return f"出错了: {e}"

# --- 5. 流式生成 (Web 用) ---
def stream_rag_response(query: str, vector_store: FAISS) -> Generator[str, None, None]:
    logger.info("正在调用 API (流式)...")
    try:
        chain = _get_rag_chain(vector_store)
        # 这里的 query 是字符串，上面的 chain 会自动把它变成字典
        for chunk in chain.stream(query):
            yield chunk
    except Exception as e:
        logger.error(f"流式失败: {e}")
        yield f"出错了: {e}"