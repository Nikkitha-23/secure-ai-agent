import streamlit as st
import requests

st.title("🔒 Secure AI Document Assistant")

question = st.text_input("Ask a question")
if st.button("Clear"):
    st.rerun()

if st.button("Ask"):

    with st.spinner("Thinking..."):

        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )

        data = response.json()

        st.subheader("Answer")
        st.success(data["answer"])

        st.subheader("Sources")
        for source in data["sources"]:
            st.write("📄", source)
