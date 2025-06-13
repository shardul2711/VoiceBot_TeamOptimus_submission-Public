import streamlit as st
from modules.asr_module import speech_to_text
from modules.response_to_speech import speak_text
from modules.response_gen import build_qa_chain, load_vectorstore
from modules.utils import (
    fetch_all_sessions,
    fetch_chat_history,
    save_conversation,
    create_new_session
)

def main():
    st.set_page_config(page_title="Voice QA Bot", layout="centered")
    st.title("🎤 Voice-Based QA System")
    st.write("Ask your questions through voice!")

    if 'current_session_id' not in st.session_state:
        st.session_state['current_session_id'] = 1
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    with st.sidebar:
        st.header("📁 Sessions")
        all_sessions = fetch_all_sessions()
        for session_id in all_sessions:
            if st.button(f"Session {session_id}", key=f"session_{session_id}"):
                st.session_state['current_session_id'] = session_id
                st.session_state['chat_history'] = fetch_chat_history(session_id)
                st.rerun()
        if st.button("➕ New Session"):
            new_session_id = create_new_session()
            st.session_state['current_session_id'] = new_session_id
            st.session_state['chat_history'] = []
            st.experimental_rerun()

    st.subheader(f"Session {st.session_state['current_session_id']}")
    for chat in st.session_state['chat_history']:
        st.write(f"**You:** {chat['user_query']}")
        st.write(f"**Bot:** {chat['bot_response']}")
        st.write("---")

    if st.button("🎙️ Speak Your Question"):
        user_query = speech_to_text()
        if user_query:
            result = qa_chain({
                "question": user_query,
                "chat_history": [(chat['user_query'], chat['bot_response']) for chat in st.session_state['chat_history']]
            })
            bot_response = result["answer"]
            save_conversation(st.session_state['current_session_id'], user_query, bot_response)
            st.session_state['chat_history'].append({
                "user_query": user_query,
                "bot_response": bot_response
            })
            speak_text(bot_response)
            st.rerun()

if __name__ == "__main__":
    main()
