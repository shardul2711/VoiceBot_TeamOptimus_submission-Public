import os
import json
from dotenv import load_dotenv  # ✅ Import for loading .env files
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# ===============================
# Load environment variables
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VECTORSTORE_DIR = "faiss_index"

# ===============================
# Load PDFs from Folder
# ===============================
def load_pdfs_from_folder(folder_path):
    all_pages = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_pages.extend(pages)
            print(f"Loaded: {file} ({len(pages)} pages)")
    return all_pages

# ===============================
# Text Splitting
# ===============================
def split_text(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    return docs

# ===============================
# Build or Load Vector Store
# ===============================
def get_vectorstore(folder_path):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(VECTORSTORE_DIR):
        print("\nLoading existing vector store...")
        vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("\nCreating vector store from PDFs...")
        pages = load_pdfs_from_folder(folder_path)
        if not pages:
            print("No PDFs found in the folder.")
            exit()
        documents = split_text(pages)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_DIR)
        print("Vector store created and saved locally.")

    return vectorstore

# ===============================
# Build QA Chain
# ===============================
def build_qa_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,  # ✅ Loaded from .env
        model_name="llama3-8b-8192"  # Recommended Groq model
    )
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    return qa_chain

# ===============================
# Run QA Chat
# ===============================
def run_qa():
    folder_path = r"ContextFiles\Documents"  # Folder where PDFs are stored
    vectorstore = get_vectorstore(folder_path)

    print("\nBuilding QA system...")
    qa_chain = build_qa_chain(vectorstore)

    print("\nQA system ready! You can now ask questions based on the uploaded PDFs.")
    print("Type 'exit' to end the session.")

    chat_history = []

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            print("Exiting the QA session. Goodbye!")
            break

        result = qa_chain({"question": query, "chat_history": chat_history})
        print("\nAnswer:", result["answer"])
        chat_history.append((query, result["answer"]))

if __name__ == "__main__":
    run_qa()
