import speech_recognition as sr
import os
from gtts import gTTS
import pygame
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import getpass

# Load environment variables from .env file
load_dotenv()

# Ensure API key is set in the environment
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

from google.ai.generativelanguage_v1beta import GenerativeServiceClient

def list_models():
    client = GenerativeServiceClient()
    models = client.list_models()
    for model in models:
        print(f"Model ID: {model.name}, Description: {model.description}")

list_models()