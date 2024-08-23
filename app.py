import streamlit as st
import os
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

# Configuring the API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found. Please set it in the .env file.")
    st.stop()

# Initialize the Google Gemini LLM model with explicit model specification
gemini_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-pro")

def generate_response(text):
    """Send text to Google Gemini LLM and get a response."""
    messages = [{"role": "user", "content": text}]
    response = gemini_model.invoke(messages)
    gemini_reply = response.content
    return gemini_reply

def text_to_speech(text):
    """Convert text to speech using gTTS and return the audio file."""
    tts = gTTS(text)
    # Use a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        temp_audio_file.seek(0)
        return temp_audio_file.name

def listen_to_speech():
    """Capture user's speech and convert it to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        #st.write("Listening...")
        audio = recognizer.listen(source)
        
    try:
        user_text = recognizer.recognize_google(audio)
        return user_text
    except sr.UnknownValueError:
        st.error("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None

def main():
    """Main function for Streamlit app."""
    st.title("Voice Interaction with LLM")

    st.sidebar.header("Description")
    st.sidebar.write("This Streamlit app enables interactive voice communication with Google's Gemini LLM. Users speak their questions, and the app captures the speech, converts it to text, sends it to the LLM for a response, and then converts the LLM's response back to speech which plays automatically.")
    st.sidebar.header("Technical Highlights")
    st.sidebar.write("Voice Input: Uses `speech_recognition` to capture and transcribe user speech.")
    st.sidebar.write("LLM Integration: Connects to Google Gemini LLM via `langchain_google_genai` for generating text responses.")
    st.sidebar.write("Text-to-Speech: Converts LLM responses to audio using `gTTS`.")
    st.sidebar.write("Autoplay: Plays audio response automatically in Streamlit using the `st.audio` widget.")
    
    # Input section
    st.header("Speak a Question")
    if st.button("Start Listening"):
        with st.spinner("Listening..."):
            user_input = listen_to_speech()
            
            if user_input:
                st.write(f"You said: {user_input}")

                # Generate LLM response
                gemini_response = generate_response(user_input)
                
                # Display the LLM response
                st.subheader("LLM Response:")
                st.write(gemini_response)
                
                # Convert response to speech
                audio_file = text_to_speech(gemini_response)
                
                # Automatically play the audio
                st.audio(audio_file, format="audio/mp3", autoplay=True)

                # Clean up temporary file
                os.remove(audio_file)
            else:
                st.error("No speech detected. Please try again.")

if __name__ == "__main__":
    main()
