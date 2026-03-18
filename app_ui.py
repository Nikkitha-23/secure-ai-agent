import streamlit as st
import requests

st.title("🔒 Secure AI Document Assistant")

question = st.text_input("Ask a question")

if st.button("Clear"):
    st.rerun()

if st.button("Ask"):
    if not question.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question},
                    timeout=60
                )
                data = response.json()

                st.subheader("Answer")
                st.success(data["answer"])

                st.subheader("Sources")
                for source in data.get("sources", []):
                    st.write("📄", source)

                st.caption(f"🔍 Search type: {data.get('search_type', 'unknown')}")
                st.caption(f"✏️ Rewritten query: {data.get('rewritten_query', question)}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                