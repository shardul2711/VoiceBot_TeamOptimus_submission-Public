# api_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
from dotenv import load_dotenv
from modules.supabase_client import supabase, save_conversation, get_conversation_history
from modules.asr_module import listen_to_user, speak_text, detect_language
from modules.vector import initialize_vector_db_for_session
from langchain_ollama import OllamaLLM
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
model = OllamaLLM(model="llama3.2")

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
# Data Models
class AssistantCreate(BaseModel):
    user_id: str
    name: str
    provider: str = "openai"
    model: str = "gpt4o"
    voice_provider: str = "deepgram"
    voice_model: str = "asteria"
    first_message: str
    system_prompt: str
    file: Optional[UploadFile] = None

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
    file_url: Optional[str]
    created_at: datetime

# Assistant Endpoints
@app.post("/assistants/create", response_model=AssistantResponse)
async def create_assistant(
    user_id: str,
    name: str,
    first_message: str,
    system_prompt: str,
    provider: str = "openai",
    model: str = "gpt4o",
    voice_provider: str = "deepgram",
    voice_model: str = "asteria",
    file: Optional[UploadFile] = None
):
    """Create a new assistant matching the exact schema"""
    try:
        assistant_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        file_url = None

        # Handle file upload if present
        if file:
            file_ext = os.path.splitext(file.filename)[1]
            file_path = f"assistant-files/uploads/{assistant_id}{file_ext}"
            
            # Upload to Supabase Storage
            with file.file as f:
                supabase.storage.from_("assistant-files").upload(
                    path=file_path,
                    file=f.read(),
                    file_options={"content-type": file.content_type}
                )
            
            # Get public URL
            file_url = supabase.storage.from_("assistant-files").get_public_url(file_path)

        # Insert into assistants table
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
            "file_url": file_url,
            "created_at": created_at
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
            "file_url": file_url,
            "created_at": created_at
        }

    except Exception as e:
        raise HTTPException(500, f"Error creating assistant: {str(e)}")

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

@app.post("/chat")
async def chat_with_agent(chat_input: ChatInput):
    """Chat endpoint - all fields from API"""
    try:
        # Format session ID correctly
        session_path = f"session_{chat_input.session_id}"

        # Initialize retriever with formatted session path
        retriever = initialize_vector_db_for_session(session_path)
        docs = retriever.invoke(chat_input.user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Get response
        response = chain.invoke({
            "context": context,
            "question": chat_input.user_query
        })

        # Store in Supabase
        supabase.table("chat_history").insert({
            "session_id": chat_input.session_id,
            "user_query": chat_input.user_query,
            "bot_response": str(response),
            "assistant_id": chat_input.assistant_id
        }).execute()

        return {"response": str(response)}

    except Exception as e:
        raise HTTPException(500, f"Error in chat: {str(e)}")


@app.post("/talk")
async def talk_with_agent(file: UploadFile = File(...), session_id: str = "session_1"):
    """Talk with agent using audio"""
    try:
        # Save temporary audio file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Transcribe
        transcription = listen_to_user(audio_file=temp_path)
        if not transcription:
            raise HTTPException(400, "Could not transcribe audio")
        
        # Get response
        retriever = initialize_vector_db_for_session(session_id)
        docs = retriever.invoke(transcription)
        combined_docs = "\n\n".join([doc.page_content for doc in docs])
        response = chain.invoke({
            "context": combined_docs,
            "question": transcription
        })
        
        # Save to database
        session_number = int(session_id.replace("session_", ""))
        supabase.table("chat_history").insert({
            "assistant_id": "lenden_assistant",
            "session_id": session_number,
            "user_query": transcription,
            "bot_response": response
        }).execute()
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "transcription": transcription,
            "response": response,
            "language": detect_language(transcription)
        }
    except Exception as e:
        raise HTTPException(500, f"Error in audio processing: {str(e)}")

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