# core.py (Day 7 终极修正版)
import os
import logging
from typing import List, Generator

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
# from langchain_core.runnables import RunnablePassthrough
# from openai import APIConnectionError, RateLimitError 

# 确保 config.py 和 prompts.py 存在
from config import MODEL_API_KEY, MODEL_BASE_URL, MODEL_NAME, VECTOR_SEARCH_K
from prompts import get_rag_prompt

logger = logging.getLogger(__name__)

# --- 1. 加载函数 (保留了 chunk_size=1000 的优化) ---
def load_and_split_document(file_path: str) -> List[Document]:
    logger.info(f"正在加载文档: {file_path}")
    from langchain_community.document_loaders import CSVLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 1. 获取文件扩展名 (例如 .pdf, .txt)
    ext = os.path.splitext(file_path)[1].lower()
    
    # 2. 工厂模式：根据后缀选择加载器
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif ext == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(file_path)
    # 🌟 【新增】：处理 CSV 文件
    elif ext == ".csv":
        # CSVLoader 会自动把每一行变成 "列名: 值" 的格式，不需要再用 Splitter 切分了！
        loader = CSVLoader(file_path, encoding="utf-8")
        pages = loader.load() # CSV 直接 load 出来就是完美的片段
        logger.info(f"CSV 加载完成，共 {len(pages)} 行数据片段")
        return pages # ⚠️ 如果是 CSV，我们直接 return pages，不让剪刀（Splitter）去碰它！
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    # 3. 加载文档
    pages = loader.load_and_split()
    
    # 4. 切分文档 (保持 Day 7 的 1000/300 配置)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300, 
        length_function=len, 
        add_start_index=True,
    )
    # texts = text_splitter.create_documents([page.page_content for page in pages])
    # 🌟 【修复 Bug】：不要用 create_documents 提取纯文本
    # 🌟 改用 split_documents，它会自动继承原有 Document 的 metadata（包含文件名）
    texts = text_splitter.split_documents(pages)
    logger.info(f"文档切分完成，共 {len(texts)} 个片段")
    return texts

# --- 2. 建库函数 ---
# core.py 的中间部分

# 定义一个专门存数据的文件夹
FAISS_DB_DIR = "faiss_index"

# 🌟 改造：加上 save_name 参数，建完库直接存硬盘
def build_vector_store(documents: List[Document], save_name: str) -> FAISS:
    logger.info("正在构建向量数据库...")
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    import os
    
    # 确保保存目录存在
    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    db = FAISS.from_documents(documents, embeddings)
    
    # 将向量库保存到本地文件夹
    save_path = os.path.join(FAISS_DB_DIR, save_name)
    db.save_local(save_path)
    logger.info(f"向量数据库已永久保存至: {save_path}")
    
    return db

# 🌟 新增：从硬盘读取数据库的函数
def load_vector_store(save_name: str):
    import os
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    
    save_path = os.path.join(FAISS_DB_DIR, save_name)
    
    # 如果本地没有这个文件，就返回 None
    if not os.path.exists(save_path):
        return None
        
    logger.info(f"✨ 发现本地缓存，正在加载: {save_path}")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    # allow_dangerous_deserialization=True 是必须加的安全确认，表示信任本地文件
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

# --- 3. 内部流水线组装 (这里修复了报错的核心) ---
def _get_rag_chain(vector_store: FAISS, prompt_template: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})
    
    llm = ChatOpenAI(
        temperature=0.3, # 稍微调高一点点，让幽默人设能发挥
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        model=MODEL_NAME,
    )
    
    # 🌟 核心修改：使用传入的模板文本
    prompt = get_rag_prompt(prompt_template)
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
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
def stream_rag_response(query: str, vector_store: FAISS, chat_history: List[dict], prompt_template: str) -> Generator[str, None, None]:
    logger.info("正在调用 API (流式)...")
    try:
        # 把 template 传给内部函数
        chain = _get_rag_chain(vector_store, prompt_template)
        
        for chunk in chain.stream({
            "question": query,
            "chat_history": chat_history
        }):
            yield chunk
    except Exception as e:
        logger.error(f"流式失败: {e}")
        yield f"出错了: {e}"