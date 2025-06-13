import speech_recognition as sr
import streamlit as st

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Speech recognition service error")
    return ""
