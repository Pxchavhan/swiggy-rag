import os
import shutil
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/swiggy-annual-report.pdf"
INDEX_DIR = "faiss_index"


def load_pdf_pymupdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append(Document(page_content=text, metadata={"page": i + 1}))
    return pages

def build_faiss_index(pdf_path: str = PDF_PATH, index_dir: str = INDEX_DIR):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    pages = load_pdf_pymupdf(pdf_path)

    # keep only non-empty pages
    pages = [p for p in pages if p.page_content and p.page_content.strip()]
    if not pages:
        raise RuntimeError(
            "No extractable text found in the PDF. If it's scanned, OCR will be required."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["source"] = os.path.basename(pdf_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)

    print(f"✅ Non-empty pages extracted: {len(pages)}")
    print(f"✅ Built FAISS index with {len(chunks)} chunks")
    print(f"✅ Saved to: {index_dir}/")

if __name__ == "__main__":
    build_faiss_index()