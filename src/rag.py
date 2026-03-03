import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.prompts import SYSTEM_PROMPT, USER_PROMPT

INDEX_DIR = "faiss_index"


def _ensure_groq_key_loaded():
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    load_dotenv()


def load_db(index_dir: str = INDEX_DIR):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_context(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page", "NA")
        text = " ".join(d.page_content.split())
        parts.append(f"[Page {page}] {text}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 6) -> str:
    _ensure_groq_key_loaded()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found.")

    db = load_db()

    docs_scores = db.similarity_search_with_score(question, k=k)

    if not docs_scores:
        return "I can't find this in the Swiggy Annual Report."

    docs = [d for d, _ in docs_scores]
    scores = [s for _, s in docs_scores]

    # Distance threshold (lower is better in FAISS)
    if min(scores) > 1.1:
        return "I can't find this in the Swiggy Annual Report."

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    msg = prompt.format_messages(
        question=question,
        context=context
    )

    response = llm.invoke(msg)

    return response.content