import streamlit as st
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from supabase import create_client, Client

# ===============================
# Environment Setup
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

VECTORSTORE_DIR = "faiss_index"
folder_path = r"ContextFiles\Documents"
SESSION_ID = 1

# ===============================
# Voice to Text
# ===============================
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return ""
        except sr.RequestError:
            st.error("Speech recognition service error")
            return ""

# ===============================
# Text to Voice
# ===============================
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ===============================
# Load Vector Store
# ===============================
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# ===============================
# Build QA Chain
# ===============================
def build_qa_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    return qa_chain

# ===============================
# Save to Supabase
# ===============================
def save_conversation(user_query, bot_response):
    response = supabase.table("chat_history").insert({
        "session_id": SESSION_ID,
        "user_query": user_query,
        "bot_response": bot_response
    }).execute()
    st.write("Supabase response:", response)

# ===============================
# Fetch Recent Questions
# ===============================
def fetch_recent_questions(session_id, limit=7):
    response = supabase.table("chat_history").select("*").eq("session_id", session_id).order("id", desc=True).limit(limit).execute()
    data = response.data
    if data:
        return list(reversed(data))  # Show oldest first
    return []

# ===============================
# Streamlit UI
# ===============================
def main():
    st.set_page_config(page_title="Voice QA Bot", layout="centered")
    st.title("🎤 Voice-Based QA System")
    st.write("Ask your questions through voice!")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    # Sidebar - Display recent questions
    with st.sidebar:
        st.header("📜 Recent Questions")
        recent_conversations = fetch_recent_questions(session_id=SESSION_ID)
        if recent_conversations:
            for convo in recent_conversations:
                st.write(f"• {convo['user_query']}")
        else:
            st.write("No questions found.")

    if st.button("🎙️ Speak Your Question"):
        user_query = speech_to_text()
        if user_query:
            result = qa_chain({"question": user_query, "chat_history": st.session_state['chat_history']})
            bot_response = result["answer"]

            # Display
            st.write(f"**You:** {user_query}")
            st.write(f"**Bot:** {bot_response}")

            # Voice Response
            speak_text(bot_response)

            # Save to Database
            save_conversation(user_query, bot_response)

            # Update chat history
            st.session_state['chat_history'].append((user_query, bot_response))

    if st.button("🔁 Reset Session"):
        st.session_state['chat_history'] = []
        st.success("Session reset successfully!")

if __name__ == "__main__":
    main()
