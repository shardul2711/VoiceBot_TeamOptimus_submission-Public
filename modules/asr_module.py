# asr_module.py

import io
from pydub import AudioSegment
import simpleaudio as sa
import speech_recognition as sr
from murf import Murf
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from modules.vector import initialize_vector_db_for_session
from modules.supabase_client import save_conversation

MURF_API_KEY = "ap2_1aa2088b-4bf0-4292-9262-fb4f4284a3ff"  # Replace with your actual API key

def detect_language(text):
    # Naive Hindi detector
    return any('\u0900' <= ch <= '\u097F' for ch in text) or any(
        word in text.lower() for word in ['mera','mujhe','naam','kaise', 'hai', 'kya', 'kaun', 'aap', 'madat', 'ji', 'bataiye']
    )

from gtts import gTTS
import pyttsx3

engine = pyttsx3.init()

def speak_text(text, lang="en"):
    print("🤖 Bot:", text)
    print("🤖 Language Detected:", lang)

    if lang == "hi":
        # Generate Hindi audio with gTTS
        tts = gTTS(text=text, lang='hi')
        tts.save("response_hi.mp3")

        # Convert MP3 to WAV
        sound = AudioSegment.from_mp3("response_hi.mp3")
        sound.export("response_hi.wav", format="wav")

        # Play the WAV
        wave_obj = sa.WaveObject.from_wave_file("response_hi.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    else:
        # Use Murf or pyttsx3 for English
        murf_client = Murf(api_key=MURF_API_KEY)
        audio_stream = murf_client.text_to_speech.stream(text=text, voice_id="en-IN-rohan")
        audio_data = b''.join(audio_stream)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")

        play_obj = sa.play_buffer(
            audio_segment.raw_data,
            num_channels=audio_segment.channels,
            bytes_per_sample=audio_segment.sample_width,
            sample_rate=audio_segment.frame_rate
        )
        play_obj.wait_done()


def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Speak now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("🗣️ You said:", text)
            return text
        except Exception as e:
            print("❌ Error recognizing speech:", e)
            return None

def run_voice_qa(session_id):
    retriever = initialize_vector_db_for_session(session_id)
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

    chat_history = []

    print("\n🧠 Voice Assistant is ready. Say 'exit' to quit.\n")
    speak_text("Voice Assistant is ready. Say something.")

    while True:
        user_query = listen_to_user()
        if not user_query:
            continue

        if user_query.lower() in ['exit', 'quit', 'stop']:
            speak_text("Goodbye!", lang="en")
            print("👋 Exiting.")
            break

        # Detect user language
        lang = "hi" if detect_language(user_query) else "en"

        # Query vector DB and LLM
        docs = retriever.invoke(user_query)
        combined_docs = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"context": combined_docs, "question": user_query})
        print("🤖 Answer:", result)

        # Speak in appropriate language
        speak_text(result, lang=lang)

        save_conversation(session_id, user_query, result)
        chat_history.append((user_query, result))
