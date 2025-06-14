# api_server.py
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
from dotenv import load_dotenv
from modules.supabase_client import supabase
from modules.asr_module import listen_to_user, speak_text, detect_language
from modules.vector import initialize_vector_db_for_session
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import uuid
from modules.sentiment_analysis import analyze_session_sentiment

# Load environment
load_dotenv()

app = FastAPI(title="LenDenClub VoiceBot API", version="1.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM and prompt from your main.py
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

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
chain = prompt | model

# Data Models
class ChatInput(BaseModel):
    assistant_id: str = "lenden_assistant"  # Default for your use case
    session_id: str
    user_query: str

class SessionCreate(BaseModel):
    assistant_id: str = "lenden_assistant"
    session_id: str

class TestDataInput(BaseModel):
    assistant_id: str = "lenden_assistant"
    questions: List[str]

# Helper Functions
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
        with open(os.path.join(upload_path, file.filename), "wb") as f:
            f.write(file.file.read())

# Endpoint Implementations
# Update your AssistantCreate model to accept multiple files
class AssistantCreate(BaseModel):
    user_id: str
    name: str
    provider: str = "groq"
    model: str = "llama3-8b-8192"
    voice_provider: str = "MURF"
    voice_model: str = "en-IN-rohan"
    first_message: str
    system_prompt: str
    files: List[UploadFile] = File(default=None)  # Changed from single file to list

class AssistantResponse(BaseModel):
    assistant_id: str
    user_id: str
    name: str
    provider: str
    model: str
    voice_provider: str
    voice_model: str
    first_message: str
    system_prompt: str
    file_urls: List[str] = []
    created_at: datetime

@app.post("/assistants/create", response_model=AssistantResponse)
async def create_assistant(
    user_id: str = Form(...),
    name: str = Form(...),
    first_message: str = Form(...),
    system_prompt: str = Form(...),
    provider: str = Form("openai"),
    model: str = Form("gpt4o"),
    voice_provider: str = Form("deepgram"),
    voice_model: str = Form("asteria"),
    files: List[UploadFile] = File(None)
):
    """Create a new assistant with multiple files and vector DB storage"""
    try:
        assistant_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        file_urls = []

        # 1. Create Context directory structure
        assistant_dir = f"Context/assistant_{assistant_id}"
        docs_dir = f"{assistant_dir}/docs"
        db_dir = f"{assistant_dir}/db"
        
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)

        # 2. Handle multiple file uploads
        if files:
            for file in files:
                file_ext = os.path.splitext(file.filename)[1]
                
                # Save to local docs directory
                local_file_path = f"{docs_dir}/{file.filename}"
                with open(local_file_path, "wb") as f:
                    f.write(await file.read())
                
                # Upload to Supabase Storage
                supabase_path = f"assistant-files/uploads/{assistant_id}/{file.filename}"
                with open(local_file_path, "rb") as f:
                    supabase.storage.from_("assistant-files").upload(
                        path=supabase_path,
                        file=f.read(),
                        file_options={"content-type": file.content_type}
                    )
                file_url = supabase.storage.from_("assistant-files").get_public_url(supabase_path)
                file_urls.append(file_url)

            # Initialize vector DB with all uploaded files
            initialize_vector_db_for_session(f"assistant_{assistant_id}")

        # 3. Create assistant record
        supabase.table("assistants").insert({
            "assistant_id": assistant_id,
            "user_id": user_id,
            "name": name,
            "provider": provider,
            "model": model,
            "voice_provider": voice_provider,
            "voice_model": voice_model,
            "first_message": first_message,
            "system_prompt": system_prompt,
            "file_urls": file_urls,  # Now stores list of URLs
            "created_at": created_at,
            "vector_db_path": db_dir
        }).execute()

        return {
            "assistant_id": assistant_id,
            "user_id": user_id,
            "name": name,
            "provider": provider,
            "model": model,
            "voice_provider": voice_provider,
            "voice_model": voice_model,
            "first_message": first_message,
            "system_prompt": system_prompt,
            "file_urls": file_urls,  # Changed to return array
            "created_at": created_at,
            "vector_db_path": db_dir
        }

    except Exception as e:
        raise HTTPException(500, f"Error creating assistant: {str(e)}")

@app.get("/assistants/{user_id}")
async def get_assistants_by_user(user_id: str):
    """Get all assistants for a specific user"""
    try:
        res = supabase.table("assistants")\
                  .select("*")\
                  .eq("user_id", user_id)\
                  .execute()
        # Return empty array if no data or data is None
        return {"assistants": res.data if res.data is not None else []}
    except Exception as e:
        raise HTTPException(404, f"Error fetching assistants: {str(e)}")
    
@app.get("/assistants/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(assistant_id: str):
    """Get assistant details"""
    try:
        res = supabase.table("assistants")\
                  .select("*")\
                  .eq("assistant_id", assistant_id)\
                  .single()\
                  .execute()
        return res.data
    except Exception as e:
        raise HTTPException(404, f"Assistant not found: {str(e)}")
# API Endpoints
@app.get("/sessions/{assistant_id}")
async def get_sessions(assistant_id: str):
    """Get existing sessions for an assistant"""
    try:
        res = supabase.table("chat_history")\
                    .select("session_id")\
                    .eq("assistant_id", assistant_id)\
                    .execute()
        
        # Get distinct session_ids
        session_ids = list({item['session_id'] for item in res.data})
        return {"sessions": session_ids}
    except Exception as e:
        raise HTTPException(500, f"Error fetching sessions: {str(e)}")

@app.post("/sessions/create")
async def create_session(data: SessionCreate):
    """Create new session - all fields from API"""
    try:
        supabase.table("chat_history").insert({
            "session_id": data.session_id,
            "assistant_id": data.assistant_id,
            "user_query": "Session created",
            "bot_response": "New session started"
        }).execute()
        return {"message": "Session created", "session_id": data.session_id}
    except Exception as e:
        raise HTTPException(500, f"Error creating session: {str(e)}")

@app.get("/history/{assistant_id}/{session_id}")
async def get_history(assistant_id: str, session_id: str):
    """Get chat history - all IDs from API"""
    try:
        res = supabase.table("chat_history")\
                    .select("*")\
                    .eq("assistant_id", assistant_id)\
                    .eq("session_id", session_id)\
                    .order("created_at")\
                    .execute()
        return {"history": res.data}
    except Exception as e:
        raise HTTPException(500, f"Error fetching history: {str(e)}")

from fastapi import Path

@app.post("/chat/{assistant_id}/{session_id}")
async def chat_with_agent(assistant_id: str = Path(...), session_id: str = Path(...), chat_input: ChatInput = None):
    """Chat endpoint scoped to assistant and session ID"""
    try:
        # Use assistant_id for vector DB context
        assistant_vector_db_path = f"assistant_{assistant_id}"
        
        # Initialize retriever with assistant's vector DB
        retriever = initialize_vector_db_for_session(assistant_vector_db_path)
        docs = retriever.invoke(chat_input.user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Get response using the assistant's context
        response = chain.invoke({
            "context": context,
            "question": chat_input.user_query
        })

        # Store in Supabase
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "user_query": chat_input.user_query,
            "bot_response": str(response),
            "assistant_id": assistant_id
        }).execute()

        return {
            "response": str(response),
            "assistant_id": assistant_id,
            "session_id": session_id,
            "vector_db_used": assistant_vector_db_path
        }

    except Exception as e:
        raise HTTPException(500, f"Error in chat: {str(e)}")


@app.get("/sentiment/{assistant_id}/{session_id}")
async def get_sentiment(assistant_id: str, session_id: str):
    """Get sentiment analysis for session using your existing function"""
    try:
        # Get chat history from Supabase
        res = supabase.table("chat_history") \
                  .select("user_query, bot_response") \
                  .eq("assistant_id", assistant_id) \
                  .eq("session_id", session_id) \
                  .order("created_at") \
                  .execute()
        
        if not res.data:
            return {"sentiment": "No chat history available"}
        
        # Format data for your function
        session_data = [{
            "user": chat["user_query"],
            "bot": chat["bot_response"]
        } for chat in res.data]

        # Call your function
        _, sentiment = analyze_session_sentiment(session_id, session_data)
        
        return {
            "assistant_id": assistant_id,
            "session_id": session_id,
            "sentiment": sentiment,
            "message_count": len(session_data)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error in sentiment analysis: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# If needed Auto Reload
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "api_server:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,  # Enable auto-reload
#         reload_dirs=["."],  # Watch current directory for changes
#         reload_includes=["*.py"],  # Watch all Python files
#         reload_excludes=["*.csv", "*.pdf"]  # Exclude data files
#     )