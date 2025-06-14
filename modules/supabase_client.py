# supabase_client.py
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_conversation(session_id, user_query, bot_response):
    session_number = int(session_id.replace("session_", ""))  # Extract numeric part
    response = supabase.table("chat_history").insert({
        "session_id": session_number,
        "user_query": user_query,
        "bot_response": bot_response
    }).execute()
    return response

def get_conversation_history(session_id):
    session_number = int(session_id.replace("session_", ""))  # If session_id stored as bigint
    response = supabase.table("chat_history").select("*").eq("session_id", session_number).order("id", desc=False).execute()
    return response.data
