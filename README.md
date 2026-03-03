# Swiggy Annual Report RAG Q&A (Retrieval-Augmented Generation)

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** based Question Answering system that answers user questions **strictly using the Swiggy Annual Report**. The system retrieves relevant excerpts from the report and uses an LLM (**Groq**) to generate **context-grounded answers with page citations**. If the answer is not present in the retrieved report context, the system **refuses** to answer to prevent hallucinations.

---

## Objective
Build an AI system that can accurately answer user questions based **ONLY** on the Swiggy Annual Report (PDF) using a RAG pipeline:

- Document ingestion and chunking  
- Embedding generation  
- Vector search (semantic retrieval)  
- LLM answer generation grounded in retrieved context  
- Simple UI for question answering  

---

## Document Source (Required)

**Swiggy Annual Report (FY 2023–24)** (publicly available PDF)

**Source link:**  
https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf

---

## Architecture

### 1) PDF Ingestion
- Load Swiggy annual report PDF  
- Extract text (page-wise)  
- Split into meaningful chunks with overlap  
- Store metadata (page number, chunk id, file source)

### 2) Embeddings + Vector Store
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`  
- Store vectors in **FAISS** for fast similarity search

### 3) RAG Answering
- User asks a natural language question  
- Top-K most relevant chunks are retrieved  
- Retrieved context + question is sent to **Groq LLM**  
- Answer is generated **only from context** + **page citations**

### 4) UI
- Streamlit web UI  
- Shows final answer  
- Optional: show retrieved context  

---

## Tech Stack
- Python 3.11  
- Streamlit (UI)  
- LangChain (RAG orchestration)  
- FAISS (vector database)  
- Sentence Transformers (embeddings)  
- Groq (LLM inference)

---
## Live Demo

- Deployed app URL:
https://swiggy-rag-cuf8tq3yqt6nypzuagxpxh.streamlit.app/

## Author
- Prachi Chavhan

## GitHub:
- https://github.com/Pxchavhan/swiggy-rag

## Repository Structure
```text
swiggy-rag/
├─ data/
│  └─ swiggy_annual_report.pdf
├─ faiss_index/              # Prebuilt FAISS index (committed for deployment stability)
├─ src/
│  ├─ app.py                 # Streamlit app
│  ├─ ingest.py              # Index build script (run locally)
│  ├─ rag.py                 # Retrieval + Groq answering logic
│  └─ prompts.py             # Strict grounding prompts
├─ requirements.txt
├─ runtime.txt               # python-3.11 for Streamlit Cloud
└─ README.md