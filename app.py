import streamlit as st
import os
from core import (load_and_split_document, build_vector_store, 
                    stream_rag_response, load_vector_store, 
                    get_query_intent, rewrite_query, 
                    text_to_speech, speech_to_text)
from langchain_core.messages import AIMessage, HumanMessage
from prompts import PERSONAS
from langchain_community.chat_message_histories import SQLChatMessageHistory
import uuid

# 1 设置页面标题
st.set_page_config(page_title="xiaolei的AI知识库助手", page_icon="🤖")
st.title("xiaolei的AI问答助手")

# 2 --- 侧边栏：文件上传区 ---
with st.sidebar:
    st.header("📚 知识库构建")
    
    uploaded_files = st.file_uploader(
        "请上传文件 (支持多选:PDF, Word, TXT, MD, CSV)", 
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("清除历史与缓存"):
        # 🌟 修复 Bug 3：必须删掉 session_id，才能强行开启一段崭新的数据库记忆
        for key in ["vector_store", "current_files", "messages", "session_id"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.header("🎭 AI 人设")
    selected_persona_name = st.selectbox(
        "选择回答风格",
        options=list(PERSONAS.keys()),
        index=0
    )
    selected_prompt_text = PERSONAS[selected_persona_name]

    st.divider()
    st.header("🔊 语音设置")
    enable_tts = st.toggle("开启语音播报", value=False)

# 4. 核心逻辑：多文件处理
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    current_file_names = sorted([f.name for f in uploaded_files])
    is_new_file_combo = "current_files" not in st.session_state or st.session_state["current_files"] != current_file_names
    
    if is_new_file_combo:
        combo_name = "_".join(current_file_names)[:50] 
        with st.spinner(f"正在联合分析 {len(uploaded_files)} 个文档..."):
            try:
                vector_store = load_vector_store(combo_name)
                
                if vector_store is None:
                    st.info("检测到新的文档组合，正在构建联合知识库...")
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("data", uploaded_file.name)
                        if not os.path.exists(file_path):
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        docs = load_and_split_document(file_path)
                        all_docs.extend(docs)
                    vector_store = build_vector_store(all_docs, combo_name)
                    st.success(f"✅ 联合知识库构建完成！共融合 {len(uploaded_files)} 个文件，{len(all_docs)} 个片段。")
                else:
                    st.success(f"⚡ 发现多文档联合缓存，秒级加载成功！")

                st.session_state["vector_store"] = vector_store
                st.session_state["current_files"] = current_file_names
                
            except Exception as e:
                st.error(f"构建失败: {e}")

# ==========================================
# 🌟 结构优化：用一个独立的大 if 来控制聊天界面，避免缩进混乱
# ==========================================
if "vector_store" in st.session_state:
    st.info(f"🧠 当前联合知识库包含 {len(st.session_state['current_files'])} 个文件 (已就绪)")
    st.divider()

    # 初始化对话历史 (如果不存在的话)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # 🌟 连接本地数据库
    chat_db = SQLChatMessageHistory(
        session_id=st.session_state.session_id,
        connection_string="sqlite:///chat_history.db"
    )

    # ==========================================
    # 🌟 核心修复：建立专属的“聊天记录包厢”
    # ==========================================
    msg_container = st.container()

    # 把历史记录、欢迎卡片、清空按钮，统统装进包厢里！
    with msg_container:
        # 空状态欢迎卡片
        if len(chat_db.messages) == 0:
            st.info("👋 **欢迎来到 xiaolei 的专属 AI 空间！**\n\n你可以直接在下方输入文字，或者点击 🎙️ 麦克风与我语音交流。试试问我文档里的核心内容吧！")

        # 从数据库中读取并渲染历史记录
        for msg in chat_db.messages:
            role = "user" if msg.type == "human" else "assistant"
            avatar_icon = "🧑‍💻" if role == "user" else "⚡" 
            with st.chat_message(role, avatar=avatar_icon):
                st.markdown(msg.content)
        
        # 一键清空对话的按钮
        if len(chat_db.messages) > 0:
            _, clear_col = st.columns([8, 2])
            with clear_col:
                if st.button("🗑️ 清空对话", use_container_width=True):
                    chat_db.clear() 
                    st.rerun()      

    # ==========================================
    # 👇 下方永远是输入区（独立于包厢之外，自然沉底）
    # ==========================================
    col1, col2 = st.columns([11, 1], vertical_alignment="bottom")
    
    with col1:
        text_prompt = st.chat_input("在这个文档里找什么？")
    with col2:
        audio_value = st.audio_input("🎙️ 语音提问", label_visibility="collapsed")
    
    prompt = text_prompt
    
    if audio_value:
        with st.spinner("👂 正在努力听懂您的话..."):
            spoken_text = speech_to_text(audio_value.getvalue())
            if spoken_text:
                prompt = spoken_text
                st.toast(f"🗣️ 识别结果: {prompt}") 
            else:
                st.error("抱歉，没听清您说什么，请再试一次或使用文字输入。")
    
    # 当有新问题产生时...
    if prompt:
        # 🌟 核心修复：把未来的新提问和新回答，强制塞回楼上的“聊天记录包厢”里！
        with msg_container:
            
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            chat_db.add_user_message(prompt)    
            
            chat_history = chat_db.messages[:-1]

            with st.spinner("⚙️ 正在理解您的真实意图..."):
                standard_query = rewrite_query(prompt, chat_history)
                intent = get_query_intent(standard_query)
                print(f"🚦 [Pipeline] 原输入: '{prompt}' | 重写为: '{standard_query}' | 路由分类: {intent}")
                
            if standard_query != prompt:
                st.caption(f"*(💡 AI 已将您的输入优化为: {standard_query})*")

            if "MALICIOUS" in intent:
                with st.chat_message("assistant", avatar="⚡"):
                    warning_msg = "🛑 **系统警告**：检测到不安全或违规的指令，请求已拦截。请规范使用知识库系统！"
                    st.error(warning_msg)
                chat_db.add_ai_message(warning_msg)
                
            elif "CHITCHAT" in intent:
                with st.chat_message("assistant", avatar="⚡"):
                    warning_msg = "☕ **闲聊拦截**：我是一个专门用来查阅文档的严肃 AI。对于写诗、讲笑话或敲代码，我实在不擅长哦，我们还是聊聊文档吧！"
                    st.info(warning_msg)
                chat_db.add_ai_message(warning_msg)
                
            else:
                with st.chat_message("assistant", avatar="⚡"):
                    with st.status("🧠 正在思考中...", expanded=True) as status:
                        st.write("🔍 正在翻阅知识库寻找线索...")
                        retrieved_docs = st.session_state["vector_store"].similarity_search(standard_query, k=3)
                        
                        st.write(f"✅ 找到了 {len(retrieved_docs)} 个高度相关的片段！")
                        st.write("⚙️ 正在结合人设组织语言...")
                        status.update(label="💡 思考完毕，开始回答！", state="complete", expanded=False)
                    
                    with st.expander("📚 查看 AI 参考的原文片段 (点击展开)"):
                        for i, doc in enumerate(retrieved_docs):
                            source_name = doc.metadata.get('source', '未知来源')
                            short_name = os.path.basename(source_name)
                            st.markdown(f"**片段 {i+1}** (来自 `{short_name}`):")
                            st.info(f"{doc.page_content[:200]}...")
                    
                    response = st.write_stream(
                        stream_rag_response(
                            prompt, 
                            st.session_state["vector_store"], 
                            chat_history,
                            selected_prompt_text 
                        )
                    )
                
                chat_db.add_ai_message(response)

                if enable_tts:
                    with st.spinner("🎵 正在合成专属语音..."):
                        audio_bytes = text_to_speech(response)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")

# 如果系统里连 vector_store 都没有，显示提示语
else:
    # 🌟 【移动端优化】：注入 HTML 和 JavaScript，制作可点击的唤醒卡片
    st.markdown("""
        <div onclick="
            // 魔法 JS:在页面里寻找那个隐藏的侧边栏按钮并模拟点击
            var sidebarBtn = window.parent.document.querySelector('[data-testid=\\'collapsedControl\\']');
            if (sidebarBtn) { 
                sidebarBtn.click(); 
            } else {
                alert('侧边栏已经是展开状态啦！请直接在左侧上传文件。');
            }
        " style="
            background-color: #e8f4fd; 
            color: #0c5487; 
            padding: 16px; 
            border-radius: 8px; 
            cursor: pointer; 
            border: 1px solid #b6d4fe;
            margin-top: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        " onmouseover="this.style.backgroundColor='#d1e7dd'" onmouseout="this.style.backgroundColor='#e8f4fd'">
            👉 <strong>温馨提示：</strong> 手机端请点击左上角的 <code>&gt;</code> 符号，或者 <b>【直接点击本框】</b> 唤出侧边栏来上传文档！
        </div>
    """, unsafe_allow_html=True)