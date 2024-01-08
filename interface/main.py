import os
import time

import streamlit as st
from openai import OpenAI
import requests

import pandas as pd
from tqdm import tqdm

from llm import LLM

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'Upload'
if 'table_page' not in st.session_state:
    st.session_state['table_page'] = 0
if 'flashcard_index' not in st.session_state:
    st.session_state['flashcard_index'] = 0

llm = LLM()

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
        print("File uploaded")
        process_file()
      

def table_page():
    st.title("Ratings Table")
    if st.session_state['data'] is not None:
        st.dataframe(st.session_state['data'])
        # TODO: add pagination
    else:
        st.write("No data uploaded. Please upload a CSV file in the Upload page.")


def flashcard_page():
    st.title('Flashcards')

    if st.session_state['data'] is not None:
      flashcards = st.session_state['data'].to_dict('records')
      selected_index = st.selectbox("Choose a flashcard", range(len(flashcards)), index=st.session_state['flashcard_index'])
      st.session_state['flashcard_index'] = selected_index
      # add previous and next buttons
      # TODO: figure out why this has a 1 click lag
      col1, col2 = st.columns(2)
      with col1:
        if st.button("Previous"):
            st.session_state['flashcard_index'] = max(0, st.session_state['flashcard_index'] - 1)
      with col2:
        if st.button("Next"):
            st.session_state['flashcard_index'] = min(len(flashcards) - 1, st.session_state['flashcard_index'] + 1)

      flashcard = flashcards[selected_index]
      st.subheader(flashcard["summary"])
      st.write("Problem: " + str(flashcard["problem"]))
      st.write("Solution: " + str(flashcard["solution"]))
    else:
      st.write("No data uploaded. Please upload a CSV file in the Upload page.")
   

# ------------------ helper functions ------------------
# Function to process the uploaded CSV file
def process_file():
    uploaded_file = st.session_state['uploaded_file']
    if uploaded_file is not None:
        # Reading the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Analyze the data
        df = analyze(df)
        # Selecting only the columns that we need
        df = df[['summary', 'problem', 'solution', 'analysis', 'innovation_score', 'feasibility_score']]
        # Store the DataFrame in the session state
        st.session_state['data'] = df
        # Automatically switch to the display page after uploading 
        # TODO: figure out why this doesn't work
        st.session_state['page'] = TABLE_PAGE_NAME

def analyze(df):
  print("Analyzing...")
  # Initialize a progress bar
  st.write("Analyzing... each entry takes about 10 seconds")
  progress_bar = st.progress(0)
  length = len(df)  # length of df changes as we drop rows
  for index, row in tqdm(df.iterrows(), total=length):
    progress = index / length
    progress_bar.progress(progress)

    filter_response = llm.filter(row)
    response = filter_response['response']
    passed = filter_response['passed']
    innovation_score = filter_response['innovation_score']
    feasibility_score = filter_response['feasibility_score']
    if not passed:
        df.drop(index, inplace=True)
        continue
    df.loc[index, 'summary'] = llm.get_summary_response(row)
    df.loc[index, 'analysis'] = response
    df.loc[index, 'innovation_score'] = innovation_score
    df.loc[index, 'feasibility_score'] = feasibility_score
  progress_bar.progress(1.0)
  st.write("Done!")
  return df
