# vector.py
import os
import json
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    Docx2txtLoader
)

embeddings = OllamaEmbeddings(model="llama3.2")

def load_documents(doc_dir):
    documents = []
    for file in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".csv"):
            loader = UnstructuredCSVLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):  # Single dict
                        documents.append(Document(page_content=json.dumps(data, indent=2)))
                    elif isinstance(data, list):  # List of dicts
                        for item in data:
                            documents.append(Document(page_content=json.dumps(item, indent=2)))
                except Exception as e:
                    print(f"Error reading JSON file {file_path}: {e}")
    return documents

def initialize_vector_db_for_session(session_id):
    doc_dir = f"Context/{session_id}/docs"
    db_path = f"Context/{session_id}/db"
    os.makedirs(db_path, exist_ok=True)

    documents = load_documents(doc_dir)

    db = Chroma(
        collection_name=f"{session_id}_collection",
        persist_directory=db_path,
        embedding_function=embeddings
    )

    if documents:
        db.add_documents(documents)

    return db.as_retriever(search_kwargs={"k": 10})
