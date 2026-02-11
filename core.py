import os
from typing import List

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

# ---------------------------------------------------------
# 第一把刀：数据加载与切分逻辑
# ---------------------------------------------------------
def load_and_split_document(file_path: str, chunk_size: int = 512, chunk_overlap: int = 200) -> List[Document]:
    """加载PDF文档并进行文本切分"""
    print(f"📦 正在加载文档: {file_path}...")
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
    print(f"✅ 文档切分完成，共 {len(texts)} 个片段。")
    return texts

# ---------------------------------------------------------
# 第二把刀：向量数据库构建逻辑
# ---------------------------------------------------------
def build_vector_store(documents: List[Document]) -> FAISS:
    """将文档片段向量化并存入 FAISS 向量数据库"""
    print("🧠 正在构建向量数据库...")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", 
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 安全读取你的 API Key
    )
    db = FAISS.from_documents(documents, embeddings)
    print("✅ 向量数据库构建完成。")
    return db

# ---------------------------------------------------------
# 第三把刀：核心 RAG 生成逻辑 (LCEL)
# ---------------------------------------------------------
def generate_rag_response(query: str, vector_store: FAISS) -> str:
    """根据用户提问，从向量库检索上下文并交由大模型生成回答"""
    print(f"🤔 正在思考问题: {query}")
    
    # 获取检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 🌟 替换这里的模型初始化逻辑 🌟
    llm = ChatOpenAI(
        temperature=0.5,
        base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"), # 优先读 env，读不到用默认
        api_key=os.getenv("DOUBAO_API_KEY"),
        model="doubao-seed-1-6-251015",
    )
    
    # 定义 Prompt
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 组装 LCEL Chain
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(query)

# ---------------------------------------------------------
# 主程序入口：指挥台
# ---------------------------------------------------------
if __name__ == "__main__":
    # 确保读取了 .env 文件中的环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 这个代码块里的逻辑非常清晰，这就是“系统架构”的雏形
    pdf_path = "./data/deepseek-v3-1-4.pdf" # 确保你本地有这个文件
    
    try:
        # 第一步：处理数据
        docs = load_and_split_document(pdf_path)
        
        # 第二步：建库
        db = build_vector_store(docs)
        
        # 第三步：向 AI 提问
        user_question = "deepseek V3有多少参数？"
        answer = generate_rag_response(user_question, db)
        
        print("\n=== 最终回答 ===")
        print(answer)
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")