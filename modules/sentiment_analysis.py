import os
from collections import defaultdict
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# ===================== Load Environment =====================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===================== Fetching Chat Data =====================
def get_all_conversations():
    response = supabase.table("chat_history").select("*").order("created_at", desc=False).execute()
    return response.data

# ===================== Group by session_id =====================
def group_by_session(chat_data):
    sessions = defaultdict(list)
    for entry in chat_data:
        sessions[entry['session_id']].append({
            "user": entry.get('user_query', entry.get('user_message', '')),
            "bot": entry.get('bot_response', '')
        })
    return sessions

# ===================== LLM Setup =====================
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)  # Updated to more powerful model

sentiment_prompt = ChatPromptTemplate.from_template("""
Analyze this conversation and classify the user's overall sentiment. 
Choose ONLY ONE of these labels:
- Good (positive/happy/satisfied)
- Moderate (neutral/indifferent)
- Disappointed (mildly unhappy)
- Bad (angry/frustrated)

Conversation:
{chat_history}

Sentiment:""")

sentiment_chain = sentiment_prompt | llm

# ===================== Optional Reason Chain =====================
reason_prompt = ChatPromptTemplate.from_template("""
The user in this conversation showed dissatisfaction. 
Provide a concise one-line explanation of why they might be unhappy.

Conversation:
{chat_history}

Reason:""")
reason_chain = reason_prompt | llm

# ===================== Analyze Sentiment for a Session =====================
def analyze_session_sentiment(session_id, session_conversation):
    if len(session_conversation) < 2:  # Skip very short conversations
        return session_id, "Neutral (too short)"

    full_dialogue = ""
    for turn in session_conversation:
        full_dialogue += f"User: {turn['user']}\nBot: {turn['bot']}\n\n"

    result = sentiment_chain.invoke({"chat_history": full_dialogue})
    response = result.content.strip()
    
    # Direct sentiment extraction
    feedback = "Moderate"  # Default to neutral
    for sentiment in ["Good", "Disappointed", "Bad"]:
        if sentiment.lower() in response.lower():
            feedback = sentiment
            break

    # Add reason if negative sentiment
    if feedback in ["Disappointed", "Bad"]:
        reason_result = reason_chain.invoke({"chat_history": full_dialogue})
        reason = reason_result.content.strip()
        feedback += f" (Reason: {reason})"

    return session_id, feedback

# ===================== Run Report =====================
def run_sentiment_report():
    print("📊 Fetching chat history...")
    data = get_all_conversations()

    print("📁 Grouping by session...")
    grouped_sessions = group_by_session(data)

    print("🧠 Running sentiment analysis...\n")
    print("Session ID | Feedback")
    print("-" * 60)

    for session_id, conv in grouped_sessions.items():
        session_id, feedback = analyze_session_sentiment(session_id, conv)
        print(f"{session_id} | {feedback}")

# ===================== Entry =====================
if __name__ == "__main__":
    run_sentiment_report()