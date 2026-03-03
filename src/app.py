import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from rag import answer_question, load_db, format_context


st.set_page_config(page_title="Swiggy Annual Report RAG", layout="wide")
st.title("Swiggy Annual Report Q&A (RAG)")
st.caption("Answers are generated strictly from the Swiggy Annual Report. If not found, it will refuse.")

question = st.text_input("Ask a question:")

col1, col2 = st.columns([1, 1])
with col1:
    k = st.slider("Top-K chunks to retrieve", min_value=2, max_value=10, value=6, step=1)
with col2:
    show_context = st.toggle("Show retrieved context", value=False)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving from report..."):
        db = load_db()
        docs = db.similarity_search(question, k=k)
        ans = answer_question(question, k=k)
        ctx = format_context(docs)

    st.subheader("Answer")
    st.write(ans)

    if show_context:
        st.subheader("Retrieved Context")
        st.write(ctx)