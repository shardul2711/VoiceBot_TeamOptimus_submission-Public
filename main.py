import os
import streamlit as st
from modules.vector import initialize_vector_db_for_session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from modules.supabase_client import save_conversation, get_conversation_history
from modules.asr_module import listen_to_user, speak_text

model = OllamaLLM(model="llama3.2")

# Initial session fallback (for Supabase etc.)
st.session_state.setdefault("session_id", "session_1")

prompt_template = """
You are a "Relationship Manager" named Satyajit working at Lenden Club, you are trained to support users with Lenden Club related queries , AND NOTHING ELSE...
(remainder unchanged for brevity)
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def get_next_assistant_id(context_root="Context"):
    os.makedirs(context_root, exist_ok=True)
    existing = [d for d in os.listdir(context_root) if d.startswith("assistant_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_id = max(nums, default=0) + 1
    return f"assistant_{next_id}"

def list_assistants(context_root="Context"):
    return sorted([d for d in os.listdir(context_root) if d.startswith("assistant_")])

def save_uploaded_files(files, assistant_id):
    upload_path = f"Context/{assistant_id}/docs"
    os.makedirs(upload_path, exist_ok=True)
    for file in files:
        with open(os.path.join(upload_path, file.name), "wb") as f:
            f.write(file.read())

st.set_page_config(page_title="LenDenClub Relationship Manager")
st.title("🤝 LenDenClub Relationship Manager")

session_mode = st.radio("Choose Assistant Session Mode", ["Use Existing Assistant", "Create New Assistant"])

if session_mode == "Use Existing Assistant":
    assistants = list_assistants()
    if not assistants:
        st.warning("No existing assistant sessions found. Please create a new one.")
    else:
        selected_assistant = st.selectbox("Select an assistant session", assistants)
        if st.button("Load Assistant"):
            st.session_state["assistant_id"] = selected_assistant
            st.session_state["retriever"] = initialize_vector_db_for_session(selected_assistant)

elif session_mode == "Create New Assistant":
    uploaded_files = st.file_uploader("Upload context files (PDF, CSV, DOCX)", type=["pdf", "csv", "docx"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Create Assistant Session"):
            new_assistant_id = get_next_assistant_id()
            save_uploaded_files(uploaded_files, new_assistant_id)
            st.session_state["assistant_id"] = new_assistant_id
            st.session_state["retriever"] = initialize_vector_db_for_session(new_assistant_id)
            st.success(f"Assistant `{new_assistant_id}` created and loaded!")

# Show chat history based on session_id
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
    st.subheader(f"💬 Chat for {st.session_state.get('assistant_id', 'N/A')}")
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
                st.markdown(f"**🧑 You said:** {user_input}")
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
