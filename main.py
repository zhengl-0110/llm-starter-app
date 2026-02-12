# main.py
import argparse
import os
import sys
from core import load_and_split_document, build_vector_store, generate_rag_response

def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="我的 AI 文档问答助手")
    
    # 2. 添加命令行参数
    # -q/--query: 用户的问题 (必须提供)
    parser.add_argument(
        "-q", "--query", 
        type=str, 
        required=True, 
        help="你想问 AI 的问题"
    )
    
    # -f/--file: 文档路径 (可选，有默认值)
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        default="./data/deepseek-v3-1-4.pdf", 
        help="要分析的 PDF 文件路径"
    )
    
    # 3. 解析用户输入的参数
    args = parser.parse_args()
    
    # 4. 防御性编程：检查文件是否存在
    if not os.path.exists(args.file):
        print(f"❌ 错误：找不到文件 '{args.file}'，请检查路径是否正确。")
        sys.exit(1) # 非正常退出

    print(f"📄 正在读取文档: {args.file}")
    print(f"❓ 正在思考问题: {args.query}")
    print("-" * 30)
    
    try:
        # 调用核心逻辑
        docs = load_and_split_document(args.file)
        db = build_vector_store(docs)
        answer = generate_rag_response(args.query, db)
        
        print("\n=== 🤖 AI 回答 ===")
        print(answer)
        print("=" * 30)
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        # 在实际工程中，这里应该记录日志 (logger.error)

if __name__ == "__main__":
    main()