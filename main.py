# main.py
from core import load_and_split_document, build_vector_store, generate_rag_response

def main():
    pdf_path = "./data/deepseek-v3-1-4.pdf"
    
    try:
        docs = load_and_split_document(pdf_path)
        db = build_vector_store(docs)
        
        user_question = "deepseek V3有多少参数?"
        answer = generate_rag_response(user_question, db)
        
        print("\n=== 最终回答 ===")
        print(answer)
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

if __name__ == "__main__":
    main()