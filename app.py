import streamlit as st
import os
from core import load_and_split_document, build_vector_store, stream_rag_response, load_vector_store, get_query_intent
from langchain_core.messages import AIMessage, HumanMessage
from prompts import PERSONAS

#1 设置页面标题
st.set_page_config(page_title="小雷的 AI 知识库助手", page_icon="🤖")
st.title("🤖 小雷的 AI 智能问答助手")

#2 --- 侧边栏：文件上传区 ---
with st.sidebar:
    st.header("📚 知识库构建")
    
    # 🌟 【改造 1】：开启 accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "请上传文件 (支持多选:PDF, Word, TXT, MD, CSV)", 
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True # <--- 关键参数！
    )
    
    if st.button("清除历史与缓存"):
        for key in ["vector_store", "current_files", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # 人设选择器保持不变
    st.divider()
    st.header("🎭 AI 人设")
    selected_persona_name = st.selectbox(
        "选择回答风格",
        options=list(PERSONAS.keys()),
        index=0
    )
    selected_prompt_text = PERSONAS[selected_persona_name]

# 4. 核心逻辑：多文件处理
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    
    # 提取当前所有上传的文件名，拼成一个列表
    current_file_names = sorted([f.name for f in uploaded_files])
    
    # 🌟 【改造 2】：判断“文件组合”是否发生了变化
    is_new_file_combo = "current_files" not in st.session_state or st.session_state["current_files"] != current_file_names
    
    if is_new_file_combo:
        # 为了给这批文件做个唯一的缓存名，我们把所有文件名拼起来做个简易 Hash
        combo_name = "_".join(current_file_names)[:50] # 截取前50个字符防名字过长
        
        with st.spinner(f"正在联合分析 {len(uploaded_files)} 个文档..."):
            try:
                # 尝试读取多文档联合缓存
                vector_store = load_vector_store(combo_name)
                
                if vector_store is None:
                    st.info("检测到新的文档组合，正在构建联合知识库...")
                    all_docs = [] # 准备一个大纸箱，用来装所有切碎的纸片
                    
                    # 遍历每一个上传的文件
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("data", uploaded_file.name)
                        # 保存到硬盘
                        if not os.path.exists(file_path):
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # 切碎，并扔进大纸箱
                        docs = load_and_split_document(file_path)
                        all_docs.extend(docs) # 把列表合并
                    
                    # 把装满所有片段的大纸箱，一次性交给 FAISS 建库
                    vector_store = build_vector_store(all_docs, combo_name)
                    st.success(f"✅ 联合知识库构建完成！共融合 {len(uploaded_files)} 个文件，{len(all_docs)} 个片段。")
                else:
                    st.success(f"⚡ 发现多文档联合缓存，秒级加载成功！")

                # 更新 Session
                st.session_state["vector_store"] = vector_store
                st.session_state["current_files"] = current_file_names
                st.session_state.messages = [] 
                
            except Exception as e:
                st.error(f"构建失败: {e}")
    
    elif "vector_store" in st.session_state:
        st.info(f"🧠 当前联合知识库包含 {len(current_file_names)} 个文件 (已就绪)")

    st.divider()

    # 3. 问答交互区 
    # 初始化对话历史 (如果不存在的话)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 🌟 关键：每次刷新页面，都要把历史聊天记录重新画出来
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 处理用户输入
    prompt = st.chat_input("在这个文档里找什么？")

    
    if prompt:
        # 1. 显示用户提问
        with st.chat_message("user"):
            st.markdown(prompt)
        # 记入历史
        st.session_state.messages.append({"role": "user", "content": prompt})

      # 🌟 【Day 17 核心修改】：呼叫大模型安检员
        with st.spinner("🛡️ 安全合规检测中..."):
            intent = get_query_intent(prompt)
            # 在终端打印出来，方便我们观察大模型是怎么分类的
            print(f"🚦 [Router] 用户输入: '{prompt}' -> 分类结果: {intent}")
        
        # 针对不同的意图，走不同的分支
        if "MALICIOUS" in intent:
            with st.chat_message("assistant"):
                warning_msg = "🛑 **系统警告**：检测到不安全或违规的指令，请求已拦截。请规范使用知识库系统！"
                st.error(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            
        elif "CHITCHAT" in intent:
            with st.chat_message("assistant"):
                warning_msg = "☕ **闲聊拦截**：我是一个专门用来查阅文档的严肃 AI。对于写诗、讲笑话或敲代码，我实在不擅长哦，我们还是聊聊文档吧！"
                st.info(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            
        else: # intent == "NORMAL" 或者解析失败默认放行
            # ✅ 安检通过：执行正常的 RAG 流水线 (保持你之前的代码不动)
            if "vector_store" in st.session_state:
                with st.chat_message("assistant"):
                    with st.status("🧠 正在思考中...", expanded=True) as status:
                        st.write("🔍 正在翻阅知识库寻找线索...")
                        # ... 下面的相似度搜索和流式输出代码保持原样 ...
                        retrieved_docs = st.session_state["vector_store"].similarity_search(prompt, k=3)
                        st.write(f"✅ 找到了 {len(retrieved_docs)} 个高度相关的片段！")
                        st.write("⚙️ 正在结合人设组织语言...")
                        status.update(label="💡 思考完毕，开始回答！", state="complete", expanded=False)
                    
                    with st.expander("📚 查看 AI 参考的原文片段 (点击展开)"):
                        for i, doc in enumerate(retrieved_docs):
                            source_name = doc.metadata.get('source', '未知来源')
                            import os
                            short_name = os.path.basename(source_name)
                            st.markdown(f"**片段 {i+1}** (来自 `{short_name}`):")
                            st.info(f"{doc.page_content[:200]}...")
                    
                    # 准备传给后端的 history
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: 
                        if msg["role"] == "user":
                            from langchain_core.messages import HumanMessage
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            from langchain_core.messages import AIMessage
                            chat_history.append(AIMessage(content=msg["content"]))

                    # 流式输出
                    response = st.write_stream(
                        stream_rag_response(
                            prompt, 
                            st.session_state["vector_store"], 
                            chat_history,
                            selected_prompt_text 
                        )
                    )
                # 记入历史
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("请先等待知识库构建完成。")

else:
    st.info("👈 请先在左侧上传一个 PDF 文档开始体验。")