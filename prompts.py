# 想优化回答效果，只要改 prompts.py
from langchain_core.prompts import ChatPromptTemplate

# 基础 RAG 问答模板
RAG_SYSTEM_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_SYSTEM_TEMPLATE)