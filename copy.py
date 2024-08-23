import speech_recognition as sr
import os
from gtts import gTTS
import pygame
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google.ai.generativelanguage_v1beta import GenerativeServiceClient

# Load environment variables from .env file
load_dotenv()

# Configuring the API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

# Initialize the Google Gemini LLM model with explicit model specification
gemini_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-pro")

def listen_to_user():
    """Capture user's speech and convert it to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        
    try:
        user_text = recognizer.recognize_google(audio)
        print(f"User said: {user_text}")
        return user_text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def generate_response(text):
    """Send text to Google Gemini LLM and get a response."""
    # Create a list of messages where the user's message has the role 'user'
    messages = [{"role": "user", "content": text}]
    
    # Generate a response from the model using the invoke method
    response = gemini_model.invoke(messages)
    
    # Print the assistant's response
    gemini_reply = response.content
    print(f"Gemini says: {gemini_reply}")
    return gemini_reply

def text_to_speech(text):
    """Convert text to speech using gTTS."""
    tts = gTTS(text)
    tts.save("response.mp3")

def play_audio(file):
    """Play the audio file using pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)

def list_models():
    client = GenerativeServiceClient()
    models = client.list_models()
    for model in models:
        print("it entered in the function")
        print(f"Model ID: {model.name}, Description: {model.description}")

def main():
    """Main loop for interaction with the LLM."""
    while True:
        # Step 1: Listen to the user's voice and convert to text
        user_input = listen_to_user()
        
        if user_input is None:
            continue
        
        # Step 2: Generate a response using the Google Gemini LLM
        gemini_response = generate_response(user_input)
        
        # Step 3: Convert the LLM's response to speech
        text_to_speech(gemini_response)
        
        # Step 4: Play the response audio to the user
        play_audio("response.mp3")

        # Optionally break the loop after one interaction
        break

if __name__ == "__main__":
    main()
