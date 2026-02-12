# main.py
import argparse
import os
import sys
import logging  # <--- 新增：导入 logging 模块
from core import load_and_split_document, build_vector_store, generate_rag_response

# 1. 配置日志系统 (这是今天的核心)
# 这段代码会让日志同时输出到“控制台”和“app.log文件”
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'), # 记录到文件
        logging.StreamHandler()                           # 显示在终端
    ]
)
# 获取当前文件的 logger
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI 文档问答助手")
    parser.add_argument("-q", "--query", type=str, required=True, help="问题")
    parser.add_argument("-f", "--file", type=str, default="./data/deepseek-v3-1-4.pdf", help="文件路径")
    args = parser.parse_args()
    
    # 使用 logger 替代 print
    logger.info(f"🚀 程序启动，准备分析文件: {args.file}")

    if not os.path.exists(args.file):
        # 使用 error 级别记录错误
        logger.error(f"❌ 文件不存在: {args.file}")
        sys.exit(1)

    try:
        docs = load_and_split_document(args.file)
        db = build_vector_store(docs)
        
        logger.info(f"❓ 用户提问: {args.query}")
        answer = generate_rag_response(args.query, db)
        
        print("\n" + "="*30)
        print(f"🤖 AI 回答:\n{answer}")
        print("="*30 + "\n")
        
        logger.info("✅ 任务成功完成")
        
    except Exception as e:
        # exc_info=True 会自动把报错的详细堆栈信息（哪一行代码错了）记录下来
        logger.error("💥 程序发生致命错误", exc_info=True)

if __name__ == "__main__":
    main()