import os
import time
import dotenv

import streamlit as st
from openai import OpenAI
import requests

import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # if the key already exists in the environment variables, it will use that, otherwise it will use the .env file to get the key
if not OPENAI_API_KEY:
    dotenv.load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'Upload'
if 'uploaded_button_clicked' not in st.session_state:
  st.session_state['uploaded_button_clicked'] = False
if 'messages' not in st.session_state:
  st.session_state['messages'] = []
if 'thread' not in st.session_state:
    st.session_state['thread'] = None
if 'assistant' not in st.session_state:
    st.session_state['assistant'] = None

client = OpenAI()

# ------------------ Names ------------------
TABLE_PAGE_NAME = "Ratings Table"
UPLOAD_PAGE_NAME = "Upload"
FLASHCARD_PAGE_NAME = "Flashcards"

# ------------------ Pages ------------------
def main():
  st.sidebar.title("Navigation")
  page = st.sidebar.radio("Go to", (UPLOAD_PAGE_NAME, TABLE_PAGE_NAME, FLASHCARD_PAGE_NAME))
  st.session_state['page'] = page

  if st.session_state['page'] == UPLOAD_PAGE_NAME:
    upload_page()
  elif st.session_state['page'] == TABLE_PAGE_NAME:
    table_page()
  elif st.session_state['page'] == FLASHCARD_PAGE_NAME:
    flashcard_page()

def upload_page():
    st.title("Upload CSV File")
    st.session_state['uploaded_file'] = st.file_uploader("Upload your CSV file here", type="csv")
    if st.session_state['uploaded_file'] is not None:
        process_file()
      

def table_page():
    st.title("Ratings Table")
    if st.session_state['data'] is not None:
        st.write("Uploaded Data:")
        st.dataframe(st.session_state['data'])
    else:
        st.write("No data uploaded. Please upload a CSV file in the Upload page.")

def flashcard_page():
   pass
   

# ------------------ helper functions ------------------
# Function to process the uploaded CSV file
def process_file():
    uploaded_file = st.session_state['uploaded_file']
    if uploaded_file is not None:
        # Reading the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Selecting only the columns that we need
        df = df[['problem', 'solution']]
        # Store the DataFrame in the session state
        st.session_state['data'] = df
        # Automatically switch to the display page after uploading 
        # TODO: figure out why this doesn't work
        st.session_state['page'] = TABLE_PAGE_NAME
