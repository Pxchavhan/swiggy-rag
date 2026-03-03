import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/Annual-Report.pdf"
INDEX_DIR = "faiss_index"

def build_faiss_index(pdf_path: str = PDF_PATH, index_dir: str = INDEX_DIR):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # each item has metadata including 'page'

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = i
        d.metadata["source"] = os.path.basename(pdf_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)

    print(f"✅ Built FAISS index with {len(chunks)} chunks")
    print(f"✅ Saved to: {index_dir}/")

if __name__ == "__main__":
    build_faiss_index()