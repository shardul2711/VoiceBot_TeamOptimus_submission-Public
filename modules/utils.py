import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_conversation(session_id, user_query, bot_response):
    response = supabase.table("chat_history").insert({
        "session_id": session_id,
        "user_query": user_query,
        "bot_response": bot_response
    }).execute()
    return response

def fetch_all_sessions():
    response = supabase.table("chat_history").select("session_id").order("session_id").execute()
    if response.data:
        return list({item['session_id'] for item in response.data})
    return []

def fetch_chat_history(session_id):
    response = supabase.table("chat_history").select("*").eq("session_id", session_id).order("id").execute()
    return response.data if response.data else []

def create_new_session():
    sessions = fetch_all_sessions()
    return max(sessions) + 1 if sessions else 1
