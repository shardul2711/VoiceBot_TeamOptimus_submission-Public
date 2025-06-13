import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTORSTORE_DIR = "faiss_index"

def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
