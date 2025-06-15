import tkinter as tk
from tkinter import scrolledtext
import requests
import pyttsx3
import re

# Config
API_URL = "http://localhost:8000/chat/daa10dc3-07ec-48cf-bf58-594ba6c36e65/1"

# Text-to-Speech setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed
tts_engine.setProperty('volume', 1.0)  

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Create GUI window
root = tk.Tk()
root.title("Simple Chatbot")
root.geometry("600x400")

# Chat display box
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 12))
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# User input box
user_input = tk.Entry(root, font=("Arial", 12))
user_input.pack(padx=10, pady=5, fill=tk.X)

def append_message(sender, message):
    chat_display.configure(state='normal')
    chat_display.insert(tk.END, f"{sender}: {message}\n")
    chat_display.configure(state='disabled')
    chat_display.yview(tk.END)

def send_message(event=None):
    query = user_input.get().strip()
    if not query:
        return

    append_message("You", query)
    user_input.delete(0, tk.END)

    try:
        res = requests.post(API_URL, json={"user_query": query})
        if res.status_code == 200:
            data = res.json()
            bot_reply = data.get("response", "⚠️ No response")
        else:
            bot_reply = f"⚠️ Error: {res.text}"
    except Exception as e:
        bot_reply = f"⚠️ Network error: {str(e)}"

    append_message("Bot", bot_reply)
    speak(bot_reply)

# Send on button click
send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12))
send_button.pack(padx=10, pady=5)

# Bind Enter key to send
user_input.bind("<Return>", send_message)

# Run the GUI loop
root.mainloop()
