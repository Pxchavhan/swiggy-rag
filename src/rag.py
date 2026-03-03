import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from prompts import SYSTEM_PROMPT, USER_PROMPT

INDEX_DIR = "faiss_index"


def _ensure_groq_key_loaded() -> None:
    """
    Loads GROQ_API_KEY from:
    1) Streamlit secrets (deployment)
    2) .env file (local)
    3) Existing environment variables
    """
    # 1) Streamlit Cloud secrets (if running under Streamlit)
    try:
        import streamlit as st  # type: ignore
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            return
    except Exception:
        pass

    # 2) Local .env
    load_dotenv()


def load_db(index_dir: str = INDEX_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def format_context(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "NA")
        text = " ".join(d.page_content.split())
        parts.append(f"[Page {page}] {text}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 6) -> str:
    _ensure_groq_key_loaded()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to .env (local) or Streamlit Secrets (deployment)."
        )

    db = load_db()
    docs = db.similarity_search(question, k=k)
    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    msg = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msg)
    return resp.content