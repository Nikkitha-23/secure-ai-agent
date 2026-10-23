"""
app_ui.py — Professional Dark-Themed Streamlit UI for Secure AI Agent
----------------------------------------------------------------------
Redesigned Sidebar Sections:
  1. ➕ New Chat
  2. 💬 Chat History (session switching & management)
  3. 📚 Knowledge Base (PDF upload, file size, status, chunk tracking)
  4. ⚙️ Preferences (voice mode, response language, clear chat vs clear memory)
  5. 📊 Agent Status (health monitor, LLM model, tools)
"""

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="🔒 Secure AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark ChatGPT-Style Sidebar & Responsive Layout
st.markdown("""
<style>
    /* Dark Theme Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        color: #F9FAFB;
        border-right: 1px solid #374151;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F9FAFB !important;
    }
    .stButton>button {
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .status-card {
        background-color: #1F2937;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .status-online { color: #10B981; font-weight: bold; }
    .status-offline { color: #EF4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────────────
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "default": [
            {
                "role": "assistant",
                "content": "Hello! I am your **Secure AI Agent**. Ask me anything, or upload PDF documents in the Knowledge Base sidebar for hybrid RAG analysis.",
                "sources": [],
                "steps": []
            }
        ]
    }

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default"

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}  # filename -> {chunks, file_size}

if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

if "language" not in st.session_state:
    st.session_state.language = "English"


# Helper: Check Backend Health & Status
def check_backend_status():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# Ensure active session exists in chat_sessions
if st.session_state.active_session_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.active_session_id] = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Secure AI Agent")
    st.caption("Enterprise Agentic System v2.0")
    st.divider()

    # ── Section 1: New Chat ───────────────────────────────────────────────────
    st.subheader("1. ➕ New Chat")
    if st.button("✨ Start Fresh Conversation", use_container_width=True):
        new_id = f"session_{int(time.time())}"
        st.session_state.active_session_id = new_id
        st.session_state.chat_sessions[new_id] = [
            {
                "role": "assistant",
                "content": "Fresh conversation started. How can I help you?",
                "sources": [],
                "steps": []
            }
        ]
        st.rerun()

    st.divider()

    # ── Section 2: Chat History ───────────────────────────────────────────────
    st.subheader("2. 💬 Chat History")
    session_keys = list(st.session_state.chat_sessions.keys())
    if session_keys:
        selected_session = st.selectbox(
            "Select Conversation",
            options=session_keys,
            index=session_keys.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_keys else 0
        )
        if selected_session != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_session
            st.rerun()

        if len(session_keys) > 1 and st.button("🗑️ Delete Selected Session", use_container_width=True):
            del st.session_state.chat_sessions[st.session_state.active_session_id]
            st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
            st.rerun()

    st.divider()

    # ── Section 3: Knowledge Base ─────────────────────────────────────────────
    st.subheader("3. 📚 Knowledge Base")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                file_kb = round(len(file.getvalue()) / 1024, 1)
                with st.spinner(f"📥 Ingesting {file.name} ({file_kb} KB)..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload",
                            files={"file": (file.name, file.read(), "application/pdf")},
                            timeout=60
                        )
                        data = res.json()
                        if res.status_code == 200:
                            chunks = data.get("chunks", 0)
                            msg = data.get("message", "")
                            st.success(f"✅ {file.name} ({chunks} chunks)")
                            st.session_state.uploaded_files[file.name] = {
                                "file_size_kb": file_kb,
                                "chunks": chunks,
                                "status": "Indexed"
                            }
                        else:
                            st.error(f"❌ {data.get('message', 'Upload failed')}")
                    except Exception as e:
                        st.error(f"❌ Ingestion error: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Indexed Documents:**")
        for fname, info in st.session_state.uploaded_files.items():
            st.caption(f"📄 **{fname}** ({info['file_size_kb']} KB | {info['chunks']} chunks)")

    st.divider()

    # ── Section 4: Preferences ────────────────────────────────────────────────
    st.subheader("4. ⚙️ Preferences")
    st.session_state.voice_mode = st.toggle("🎤 Voice Mode", value=st.session_state.voice_mode)
    st.session_state.language = st.selectbox("Response Language", ["English", "Tamil", "Spanish", "French", "German"])

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_sessions[st.session_state.active_session_id] = [
                {"role": "assistant", "content": "Chat history cleared.", "sources": [], "steps": []}
            ]
            st.rerun()
    with col_c2:
        if st.button("🗑️ Clear Memory", use_container_width=True):
            try:
                requests.delete(f"{BACKEND_URL}/memory/clear?session_id={st.session_state.active_session_id}", timeout=5)
                st.success("Memory cleared!")
            except Exception as e:
                st.error(f"Clear failed: {e}")

    st.divider()

    # ── Section 5: Agent Status ───────────────────────────────────────────────
    st.subheader("5. 📊 Agent Status")
    health = check_backend_status()
    if health:
        st.markdown(f"""
        <div class="status-card">
            🟢 <b>Backend Status:</b> <span class="status-online">Online</span><br>
            🧠 <b>LLM Model:</b> {health.get('model', 'Llama 3.3 70B')}<br>
            📚 <b>Max Agent Steps:</b> {health.get('max_agent_steps', 5)}<br>
            🛠️ <b>Registered Tools:</b> 7 Active Tools
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            🔴 <b>Backend Status:</b> <span class="status-offline">Offline</span><br>
            ⚠️ Cannot reach FastAPI server on port 8000
        </div>
        """, unsafe_allow_html=True)


# ── Main Conversation Interface ────────────────────────────────────────────────
st.title("🔒 Secure AI Agent Chatbot")
st.caption(f"Session: `{st.session_state.active_session_id}` | Language: `{st.session_state.language}`")

current_messages = st.session_state.chat_sessions[st.session_state.active_session_id]

# Render current session messages
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        sources = [str(s).strip() for s in msg.get("sources", []) if s and str(s).strip()]
        if sources:
            with st.expander("📄 Source Attributions", expanded=False):
                for s in sources:
                    st.markdown(f"- {s}")

        steps = msg.get("steps", [])
        if steps and len(steps) > 1:
            with st.expander("🔍 Agent Execution Steps", expanded=False):
                for step in steps:
                    st.caption(f"**Step {step.get('step_num', 1)} [{step.get('tool', 'agent')}]**: {step.get('purpose', '')} ({step.get('status', 'done')})")


# Input Box
prompt = st.chat_input("Ask a question or request a task...")

if prompt:
    # Append user message
    current_messages.append({"role": "user", "content": prompt, "sources": [], "steps": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🤖 Orchestrating plan & execution..."):
            try:
                # Add language instruction if non-English
                query_text = prompt
                if st.session_state.language != "English":
                    query_text = f"Please answer in {st.session_state.language}: {prompt}"

                res = requests.post(
                    f"{BACKEND_URL}/agent",
                    json={
                        "question": query_text,
                        "session_id": st.session_state.active_session_id,
                        "user_role": "student",
                        "user_id": "default_user"
                    },
                    timeout=90
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response generated.")
                    sources = data.get("sources", [])
                    steps = data.get("steps", [])
                else:
                    answer = f"⚠️ Backend returned error status code: {res.status_code}"
                    sources = []
                    steps = []
            except Exception as e:
                answer = f"❌ Error connecting to agent backend: {str(e)}"
                sources = []
                steps = []

            st.markdown(answer)

            valid_sources = [str(s).strip() for s in sources if s and str(s).strip()]
            if valid_sources:
                with st.expander("📄 Source Attributions", expanded=True):
                    for s in valid_sources:
                        st.markdown(f"- {s}")

            if steps and len(steps) > 1:
                with st.expander("🔍 Agent Execution Steps", expanded=False):
                    for step in steps:
                        st.caption(f"**Step {step.get('step_num', 1)} [{step.get('tool', 'agent')}]**: {step.get('purpose', '')} ({step.get('status', 'done')})")

            # Voice TTS
            if st.session_state.voice_mode and answer:
                try:
                    from rag.voice import text_to_speech
                    audio_path = text_to_speech(answer[:500])
                    if audio_path:
                        with open(audio_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        os.unlink(audio_path)
                except Exception:
                    pass

            current_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "steps": steps
            })