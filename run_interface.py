import os
import streamlit as st
from modules.vector import initialize_vector_db_for_session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from modules.supabase_client import save_conversation, get_conversation_history
from modules.asr_module import listen_to_user, speak_text

model = OllamaLLM(model="llama3.2")

st.session_state.setdefault("session_id", "session_1")

prompt_template = """
You are a "Relationship Manager" named Satyajit working at Lenden Club, you are trained to support users with Lenden Club related queries , AND NOTHING ELSE. Lenden Club is a peer-to-peer (P2P) lending platforms. Your job is to help with the following three tasks for LenDenClub Customers, India's largest P2P lending platform.

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

Relevant Documents:
{context}

User Query:
{question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def get_next_session_id(context_root="Context"):
    os.makedirs(context_root, exist_ok=True)
    existing = [d for d in os.listdir(context_root) if d.startswith("session_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_id = max(nums, default=0) + 1
    return f"session_{next_id}"

def list_sessions(context_root="Context"):
    return sorted([d for d in os.listdir(context_root) if d.startswith("session_")])

def save_uploaded_files(files, session_id):
    upload_path = f"Context/{session_id}/docs"
    os.makedirs(upload_path, exist_ok=True)
    for file in files:
        with open(os.path.join(upload_path, file.name), "wb") as f:
            f.write(file.read())

st.set_page_config(page_title="LenDenClub Relationship Manager")
st.title("🤝 LenDenClub Relationship Manager")

session_mode = st.radio("Choose Session Mode", ["Use Existing Session", "Create New Session"])

if session_mode == "Use Existing Session":
    sessions = list_sessions()
    if not sessions:
        st.warning("No existing sessions found. Please create a new one.")
    else:
        selected_session = st.selectbox("Select a session", sessions)
        if st.button("Load Session"):
            st.session_state["session_id"] = selected_session
            st.session_state["retriever"] = initialize_vector_db_for_session(selected_session)

elif session_mode == "Create New Session":
    uploaded_files = st.file_uploader("Upload context files (PDF, CSV, DOCX)", type=["pdf", "csv", "docx"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Create Session"):
            new_session_id = get_next_session_id()
            save_uploaded_files(uploaded_files, new_session_id)
            st.session_state["session_id"] = new_session_id
            st.session_state["retriever"] = initialize_vector_db_for_session(new_session_id)
            st.success(f"Session `{new_session_id}` created and loaded!")

# Show chat history
st.markdown("### 🕘 Previous Chat History")
chat_history = get_conversation_history(st.session_state["session_id"])

if chat_history:
    for chat in chat_history:
        st.markdown(f"**🧑 You:** {chat['user_query']}")
        st.markdown(f"**🤖 Bot:** {chat['bot_response']}")
        st.markdown("---")
else:
    st.info("No previous chats in this session.")

# Chat Interface
if "retriever" in st.session_state:
    st.subheader(f"💬 Chat for {st.session_state['session_id']}")
    chat_mode = st.radio("Choose Input Mode", ["Text", "Voice"])

    if chat_mode == "Text":
        user_input = st.text_input("Ask a question about P2P Lending / LenDenClub")
        if user_input:
            docs = st.session_state["retriever"].invoke(user_input)
            combined_docs = "\n\n".join([doc.page_content for doc in docs])
            chain = prompt | model
            response = chain.invoke({"context": combined_docs, "question": user_input})
            st.markdown("**Response:**")
            st.write(response)
            save_conversation(st.session_state["session_id"], user_input, str(response))

    elif chat_mode == "Voice":
        if st.button("🎤 Start Voice Conversation"):
            user_input = listen_to_user()
            if user_input:
                st.markdown(f"**🧑 You said:** {user_input}")  # 👈 This line shows what the user said

                docs = st.session_state["retriever"].invoke(user_input)
                combined_docs = "\n\n".join([doc.page_content for doc in docs])
                chain = prompt | model
                response = chain.invoke({"context": combined_docs, "question": user_input})

                st.markdown("**🤖 Response:**")
                st.write(response)

                speak_text(str(response))
                save_conversation(st.session_state["session_id"], user_input, str(response))
            else:
                st.warning("Sorry, could not recognize your voice.")

