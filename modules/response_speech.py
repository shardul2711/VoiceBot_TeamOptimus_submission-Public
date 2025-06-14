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
# Environment Setup D:\Hackathon\matrix protocol\NLP\new for github\VoiceBot_TeamOptimus_submission\run_interface.py
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

VECTORSTORE_DIR = "faiss_index"

# ===============================
# Relationship Manager Agent Prompt
# ===============================
RELATIONSHIP_MANAGER_PROMPT = """
You are a "Relationship Manager" trained to support users with peer-to-peer (P2P) lending platforms. Your job is to help with the following three tasks for LenDenClub Customers, India's largest P2P lending platform.

When responding to queries about P2P lending or LenDenClub, always follow these guidelines:

1) Help with Initial Onboarding:
- Explain platform features simply
- Mention fund diversification starts from ₹100
- State max lending amount is ₹10 Lakhs
- Highlight escrow safety with ICICI Trusteeship
- Share expected returns (~11.33% p.a.)
- Explain borrower verification (600+ data points)
- Mention 95.6% on-time repayment rate

2) Explain Key Terms (simple definitions):
- P2P Lending: Direct lending between individuals via platform
- AUM (₹1,023 Cr): Total money managed by platform
- NPA (3.78%): Loans not repaid on time
- Escrow: Protected account managed by ICICI Trusteeship
- Diversification: Spreading ₹100+ across multiple loans
- EMI: Monthly installment payments
- Interest vs Returns: What borrowers pay vs lenders earn
- InstaMoney: LenDenClub's app (3Cr+ downloads)

3) Risk Management:
- Clearly state: "P2P lending carries risks"
- Mention RBI regulates the platform (NBFC-P2P)
- Explain 3.78% NPA means some loans may default
- Stress importance of diversification
- Highlight escrow protection
- Note 95.6% repayment rate
- Mention zero principal loss since launch

Always use the latest platform data (Dec 2024):
- 2Cr+ users, ₹16,011Cr total lent
- 85% personal, 15% merchant loans
- RBI registered (Innofin Solutions Pvt Ltd)
"""

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
def save_conversation(session_id, user_query, bot_response):
    response = supabase.table("chat_history").insert({
        "session_id": session_id,
        "user_query": user_query,
        "bot_response": bot_response
    }).execute()
    return response

# ===============================
# Fetch All Sessions
# ===============================
def fetch_all_sessions():
    response = supabase.table("chat_history").select("session_id").order("session_id").execute()
    if response.data:
        # Get unique session IDs
        sessions = list({item['session_id'] for item in response.data})
        return sessions
    return []

# ===============================
# Fetch Chat History for Session
# ===============================
def fetch_chat_history(session_id):
    response = supabase.table("chat_history").select("*").eq("session_id", session_id).order("id").execute()
    if response.data:
        return response.data
    return []

# ===============================
# Create New Session
# ===============================
def create_new_session():
    # Find the highest session ID and increment by 1
    sessions = fetch_all_sessions()
    if sessions:
        new_session_id = max(sessions) + 1
    else:
        new_session_id = 1
    return new_session_id

# ===============================
# Modified Main Function
# ===============================
def main():
    st.set_page_config(page_title="Voice QA Bot", layout="centered")
    st.title("🎤 Voice-Based QA System")
    st.write("Ask your questions through voice!")

    # Initialize session state
    if 'current_session_id' not in st.session_state:
        st.session_state['current_session_id'] = 1
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Load vector store and QA chain
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    # Sidebar - Session Management
    with st.sidebar:
        st.header("📁 Sessions")
        
        # Get all sessions from database
        all_sessions = fetch_all_sessions()
        
        # Display session buttons
        for session_id in all_sessions:
            if st.button(f"Session {session_id}", key=f"session_{session_id}"):
                st.session_state['current_session_id'] = session_id
                st.session_state['chat_history'] = fetch_chat_history(session_id)
                st.rerun()
        
        # Create new session button
        if st.button("➕ New Session"):
            new_session_id = create_new_session()
            st.session_state['current_session_id'] = new_session_id
            st.session_state['chat_history'] = []
            st.rerun()

    # Main Content Area
    st.subheader(f"Session {st.session_state['current_session_id']}")
    
    # Display current session chat history
    for chat in st.session_state['chat_history']:
        st.write(f"**You:** {chat['user_query']}")
        st.write(f"**Bot:** {chat['bot_response']}")
        st.write("---")
    
    # Voice input button
    if st.button("🎙️ Speak Your Question"):
        user_query = speech_to_text()
        if user_query:
            # Check if query is related to P2P/LenDenClub
            p2p_keywords = ['lendenclub', 'p2p', 'lending', 'borrow', 'interest', 'npa', 'aum', 'escrow']
            if any(keyword in user_query.lower() for keyword in p2p_keywords):
                # Use Relationship Manager mode
                llm = ChatGroq(
                    temperature=0.3,  # Slightly more creative for explanations
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama3-8b-8192"
                )
                response = llm.invoke(RELATIONSHIP_MANAGER_PROMPT + "\n\nUser Question: " + user_query)
                bot_response = response.content
            else:
                # Use normal QA chain
                result = qa_chain({
                    "question": user_query, 
                    "chat_history": [(chat['user_query'], chat['bot_response']) for chat in st.session_state['chat_history']]
                })
                bot_response = result["answer"]

            # Save to database
            save_conversation(st.session_state['current_session_id'], user_query, bot_response)
            
            # Update chat history in session state
            st.session_state['chat_history'].append({
                "user_query": user_query,
                "bot_response": bot_response
            })
            
            # Voice response
            speak_text(bot_response)
            
            # Rerun to update display
            st.rerun()

if __name__ == "__main__":
    main()