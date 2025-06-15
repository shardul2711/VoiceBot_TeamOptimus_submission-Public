from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import os
import pandas as pd
from dotenv import load_dotenv
from modules.supabase_client import supabase
from modules.vector import initialize_vector_db_for_session
from datetime import datetime
import uuid
from modules.sentiment_analysis import analyze_session_sentiment
from groq import Groq

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

# Custom Prompt Template Implementation
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class CustomPromptTemplate:
    def __init__(self, messages: List[Dict[str, str]]):
        self.messages = [Message(**msg) for msg in messages]
    
    def format(self, **kwargs) -> List[Dict[str, str]]:
        formatted_messages = []
        for msg in self.messages:
            try:
                formatted_content = msg.content.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} in prompt template")
                
            formatted_messages.append({
                'role': msg.role,
                'content': formatted_content
            })
        return formatted_messages

def create_assistant_prompt(system_prompt: str, first_message: str, context: str = "") -> CustomPromptTemplate:
    """Create a custom prompt template for an assistant"""
    full_system_prompt = f"{system_prompt}\n\nContext:\n{context}" if context else system_prompt
    return CustomPromptTemplate([
        {'role': 'system', 'content': full_system_prompt},
        {'role': 'human', 'content': "Hello!"},
        {'role': 'ai', 'content': first_message},
        {'role': 'human', 'content': "{user_input}"}
    ])

# Initialize Groq client for LLM
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(messages: List[Dict[str, str]], model: str = "llama3-8b-8192", language: str = "en") -> str:
    """Generate response enforcing the specified language"""
    try:
        # Convert our message format to Groq's expected format
        groq_messages = []
        role_mapping = {
            'system': 'system',
            'human': 'user',
            'ai': 'assistant'
        }
        
        # Add language instruction to the system prompt
        if language != "en":
            for msg in messages:
                if msg['role'] == 'system':
                    msg['content'] = f"{msg['content']}\n\nImportant: You must respond in {language} language only."
                    break
        
        for msg in messages:
            if msg['role'] not in role_mapping:
                raise ValueError(f"Invalid role: {msg['role']}")
                
            groq_messages.append({
                "role": role_mapping[msg['role']],
                "content": msg['content']
            })
        
        # For Hindi, we need to be more explicit with the instruction
        if language == "hi":
            groq_messages.append({
                "role": "user",
                "content": "कृपया हिंदी में ही उत्तर दें।"
            })
        
        chat_completion = groq_client.chat.completions.create(
            messages=groq_messages,
            model=model,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Data Models
class SessionCreate(BaseModel):
    assistant_id: str = "lenden_assistant"
    session_id: str

class TestDataInput(BaseModel):
    assistant_id: str = "lenden_assistant"
    questions: List[str]

class ChatInput(BaseModel):
    user_query: str

class AssistantCreate(BaseModel):
    user_id: str
    name: str
    provider: str = "groq"
    model: str = "llama3-8b-8192"
    voice_provider: str = "MURF"
    voice_model: str = "en-IN-rohan"
    first_message: str
    system_prompt: str
    files: List[UploadFile] = File(default=None)

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

# API Endpoints
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
            "file_urls": file_urls,
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
            "file_urls": file_urls,
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

@app.get("/sessions/{assistant_id}")
async def get_sessions(assistant_id: str):
    """Get existing sessions for an assistant"""
    try:
        res = supabase.table("chat_history")\
                    .select("session_id")\
                    .eq("assistant_id", assistant_id)\
                    .execute()
        
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

@app.post("/chat/{assistant_id}/{session_id}")
async def chat_with_agent(
    assistant_id: str = Path(...),
    session_id: str = Path(...),
    chat_input: ChatInput = None
):
    """Chat endpoint scoped to assistant and session ID"""
    try:
        if chat_input is None or not chat_input.user_query:
            raise HTTPException(status_code=400, detail="Missing user_query")

        # Fetch assistant config
        assistant = supabase.table("assistants").select("*").eq("assistant_id", assistant_id).execute()
        if not assistant.data:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        assistant_config = assistant.data[0]
        
        # Vector retrieval
        assistant_vector_db_path = f"assistant_{assistant_id}"
        retriever = initialize_vector_db_for_session(assistant_vector_db_path)
        docs = retriever.invoke(chat_input.user_query)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

        # Create and format prompt
        prompt = create_assistant_prompt(
            system_prompt=assistant_config.get("system_prompt", ""),
            first_message=assistant_config.get("first_message", ""),
            context=context
        )
        messages = prompt.format(user_input=chat_input.user_query)
        
        # Generate response
        bot_response = generate_response(messages)

        # Store conversation
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "user_query": chat_input.user_query,
            "bot_response": bot_response,
            "assistant_id": assistant_id
        }).execute()

        return {
            "response": bot_response,
            "assistant_id": assistant_id,
            "session_id": session_id,
            "vector_db_used": assistant_vector_db_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/voice-chat/{assistant_id}/{session_id}")
async def voice_chat_with_agent(
    assistant_id: str = Path(...),
    session_id: str = Path(...),
    audio_file: UploadFile = File(...),
    language: str = Form("hi")  # Default to Hindi now
):
    """Unified endpoint: audio + language → transcription → context retrieval → response"""
    try:
        # Step 0: Fetch assistant configuration
        assistant = supabase.table("assistants").select("*").eq("assistant_id", assistant_id).execute()
        if not assistant.data:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        assistant_config = assistant.data[0]
        system_prompt = assistant_config.get("system_prompt", "")
        first_message = assistant_config.get("first_message", "")
        
        # Step 1: Save uploaded audio temporarily
        temp_audio_path = f"temp_{uuid.uuid4()}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(await audio_file.read())

        # Step 2: Transcribe with Whisper (using provided language)
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file.filename, file.read()),
                model="whisper-large-v3-turbo",
                language=language,
                response_format="verbose_json",
            )
        user_query = transcription.text

        os.remove(temp_audio_path)

        # Step 3: Vector retrieval
        assistant_vector_db_path = f"assistant_{assistant_id}"
        retriever = initialize_vector_db_for_session(assistant_vector_db_path)
        docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

        # Step 4: Generate response with strong Hindi enforcement
        hindi_system_prompt = f"""
        {system_prompt}
        
        Important Instructions:
        1. You must respond in Hindi language only.
        2. Use simple Hindi words that are easy to understand.
        3. If you don't know a Hindi word, explain the concept in simple Hindi.
        4. Never respond in English.
        
        Context:
        {context}
        """
        
        prompt = create_assistant_prompt(
            system_prompt=hindi_system_prompt,
            first_message=first_message,
            context=context
        )
        messages = prompt.format(user_input=user_query)
        
        bot_response = generate_response(
            messages,
            model="llama3-8b-8192",
            language=language
        )

        # Step 5: Store in Supabase
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "user_query": user_query,
            "bot_response": bot_response,
            "assistant_id": assistant_id
        }).execute()

        return {
            "response": bot_response,
            "transcription": user_query,
            "assistant_id": assistant_id,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in voice chat: {str(e)}")

@app.get("/sentiment/{assistant_id}/{session_id}")
async def get_sentiment(assistant_id: str, session_id: str):
    """Get sentiment analysis for session"""
    try:
        res = supabase.table("chat_history") \
                  .select("user_query, bot_response") \
                  .eq("assistant_id", assistant_id) \
                  .eq("session_id", session_id) \
                  .order("created_at") \
                  .execute()
        
        if not res.data:
            return {"sentiment": "No chat history available"}
        
        session_data = [{
            "user": chat["user_query"],
            "bot": chat["bot_response"]
        } for chat in res.data]

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