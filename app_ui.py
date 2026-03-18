import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🔒 Secure AI Assistant", layout="centered")
st.title("🔒 Secure AI Document Assistant")

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False
if "recorded_question" not in st.session_state:
    st.session_state.recorded_question = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.session_id = st.text_input("Session ID", value="default")
    st.session_state.voice_mode = st.toggle("🎤 Voice Mode", value=False)

    if st.button("🗑️ Clear Memory"):
        requests.delete(f"http://127.0.0.1:8000/memory/clear?session_id={st.session_state.session_id}")
        st.success("Memory cleared!")

    st.divider()
    st.caption("Voice Mode ON — mic record பண்ணும் + answer audio-ஆ play ஆகும்")

# ── Input ──────────────────────────────────────────────────────────────────────
question = ""

if st.session_state.voice_mode:
    st.info("🎤 Voice Mode Active")
    duration = st.slider("Recording duration (seconds)", 3, 10, 5)

    if st.button("🎙️ Record & Transcribe"):
        with st.spinner("🎤 Recording..."):
            try:
                from rag.voice import record_and_transcribe
                transcribed = record_and_transcribe(duration=duration)
                if transcribed:
                    st.session_state.recorded_question = transcribed
                    st.success(f"📝 You said: {transcribed}")
                else:
                    st.error("❌ Could not transcribe audio")
            except Exception as e:
                st.error(f"❌ Voice error: {e}")

    # Show transcribed question
    if st.session_state.recorded_question:
        st.info(f"🎤 Question: {st.session_state.recorded_question}")
    question = st.session_state.recorded_question

else:
    question = st.text_input("Ask a question")

# ── Buttons ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    ask_clicked = st.button("Ask")
with col2:
    if st.button("Clear"):
        st.session_state.recorded_question = ""
        st.rerun()

# ── Ask ────────────────────────────────────────────────────────────────────────
if ask_clicked:
    if not question or not question.strip():
        st.warning("⚠️ Please enter or record a question!")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                )
                data = response.json()
                answer = data.get("answer", "No answer")

                st.subheader("💬 Answer")
                st.success(answer)

                # 🔊 Voice Output
                if st.session_state.voice_mode:
                    try:
                        from rag.voice import text_to_speech
                        with st.spinner("🔊 Generating audio..."):
                            audio_path = text_to_speech(answer[:500])
                            if audio_path:
                                with open(audio_path, "rb") as f:
                                    st.audio(f.read(), format="audio/mp3")
                                os.unlink(audio_path)
                    except Exception as e:
                        st.warning(f"⚠️ Audio generation failed: {e}")

                # Sources
                st.subheader("📄 Sources")
                for source in data.get("sources", []):
                    st.write("📄", source)

                st.caption(f"🔍 Search: {data.get('search_type', 'unknown')} | ✏️ Rewritten: {data.get('rewritten_query', question)}")

                # Clear recorded question after asking
                st.session_state.recorded_question = ""

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                