#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


#get_ipython().system('python -m spacy download en_core_web_sm')


# In[3]:


import openai
import spacy
import streamlit as st
import pandas as pd
import nltk
import PyPDF2
import docx
from collections import Counter
import re
import torch
import transformers
import pytesseract
import pdf2image
from dotenv import load_dotenv
import os
import subprocess
import sys
from textblob import TextBlob
from textblob import Word
import requests
import nltk

# In[4]:

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define a writable directory for NLTK data
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Set the custom directory for NLTK data
nltk.data.path.append(NLTK_DATA_PATH)
os.environ['NLTK_DATA'] = NLTK_DATA_PATH  # Manually set env variable

# Download required NLTK corpora
#nltk.download('punkt', download_dir=NLTK_DATA_PATH)  # Tokenization
#nltk.download('wordnet', download_dir=NLTK_DATA_PATH)  # Lemmatization
#nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)  # POS Tagging
#nltk.download('brown', download_dir=NLTK_DATA_PATH)  # Required for TextBlob

# Force load the corpora before using TextBlob
try:
    Word("hello").synsets  # Forces TextBlob to check its corpora
    _ = Spelling()  # Ensures spelling corpus is loaded
except Exception as e:
    st.error(f"TextBlob corpus loading error: {str(e)}")
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def generate_ielts_question():
    """Generate an IELTS-style writing question using GPT-4."""
    messages = [
        {"role": "system", "content": "You are an IELTS examiner. Generate a random IELTS Writing Task 1 or Task 2 question."},
        {"role": "user", "content": "Give me an IELTS Writing Task 2 question."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    return response.choices[0].message.content

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded PDF or DOCX file."""
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return None

def analyze_essay(text, prompt):
    """Performs NLP analysis and evaluates relevance to the question using OpenAI GPT."""
    doc = nlp(text)

    # Word count
    word_count = len([token.text for token in doc if token.is_alpha])

    # Sentence count
    sentence_count = len(list(doc.sents))

    # Lexical diversity
    unique_words = set([token.text.lower() for token in doc if token.is_alpha])
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0

    # Grammar mistakes (basic heuristic: count of dependency errors)
    grammar_errors = sum(1 for token in doc if token.dep_ == "punct")

    # Sentence complexity (average words per sentence)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # OpenAI GPT relevance check
    messages = [
        {"role": "system", "content": "You are an IELTS examiner. Assess the given essay based on the IELTS criteria."},
        {"role": "user", "content": f"Prompt: {prompt}\nEssay: {text}\n\nGive band scores for Task Achievement, Coherence, Lexical Resource, Grammar, and an Overall Band Score. Then provide detailed feedback on how to improve."}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    feedback_text = response.choices[0].message.content

    # Extracting scores from feedback using regex
    scores = {
        "Task Achievement": None,
        "Coherence": None,
        "Lexical Resource": None,
        "Grammar": None,
        "Overall Band Score": None
    }

    patterns = {
        "Task Achievement": r"Task Achievement.*?(\d(?:\.\d)?)",
        "Coherence": r"Coherence.*?(\d(?:\.\d)?)",
        "Lexical Resource": r"Lexical Resource.*?(\d(?:\.\d)?)",
        "Grammar": r"Grammar.*?(\d(?:\.\d)?)",
        "Overall Band Score": r"Overall Band Score.*?(\d(?:\.\d)?)"
    }

    for category, pattern in patterns.items():
        match = re.search(pattern, feedback_text, re.IGNORECASE)
        if match:
            scores[category] = float(match.group(1))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": round(lexical_diversity, 2),
        "grammar_errors": grammar_errors,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "Task Achievement": scores["Task Achievement"],
        "Coherence": scores["Coherence"],
        "Lexical Resource": scores["Lexical Resource"],
        "Grammar": scores["Grammar"],
        "Overall Band Score": scores["Overall Band Score"],
        "feedback": feedback_text  # Full feedback from OpenAI
    }

def check_grammar(text):
    url = "https://api.languagetool.org/v2/check"
    data = {
        "text": text,
        "language": "en-AU"  # Change to 'en-GB' if needed
    }
    response = requests.post(url, data=data)
    return response.json().get("matches", [])

# Streamlit UI
st.title("IELTS Writing Test")

if st.button("Generate IELTS Question"):
    ielts_question = generate_ielts_question()
    st.session_state['ielts_question'] = ielts_question

if 'ielts_question' in st.session_state:
    st.subheader("Your IELTS Question:")
    st.write(st.session_state['ielts_question'])
    
    # Input options
    user_input = st.text_area("Type your response:", height=300, key="user_input")
    uploaded_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"])

    # Live word count and spell/grammar check
    # Live word count and spell/grammar check
    if user_input:
        words = user_input.split()
        word_count = len(words)
        st.write(f"Word Count: **{word_count}**")

        # Spell checking with TextBlob (Fixed)
        blob = TextBlob(user_input)
        spelling_suggestions = []
        for word in blob.words:
            if word.spellcheck()[0][1] < 1:  # If confidence is low, suggest correction
                spelling_suggestions.append(word.correct())

        # Grammar checking using the API
        grammar_corrections = check_grammar(user_input)
        
        if spelling_suggestions:
            st.write("### Spelling Suggestions:")
            st.write(", ".join(spelling_suggestions))

        if grammar_corrections:
            st.write("### Grammar Issues:")
            for correction in grammar_corrections:
                st.write(f"**{correction['context']}** → {correction['replacements']}")

    if st.button("Submit Response"):
        essay_text = user_input if user_input.strip() else None
        if uploaded_file is not None:
            essay_text = extract_text_from_file(uploaded_file)
        
        if essay_text:
            result = analyze_essay(essay_text, st.session_state['ielts_question'])
    
            st.subheader("Assessment Result:")
            st.write(f"**Word Count:** {result['word_count']}")
            st.write(f"**Sentence Count:** {result['sentence_count']}")
            st.write(f"**Lexical Diversity:** {result['lexical_diversity']}")
            st.write(f"**Grammar Errors:** {result['grammar_errors']}")
            st.write(f"**Average Sentence Length:** {result['avg_sentence_length']} words")
        
            # Display individual band scores
            st.subheader("Band Scores:")
            st.write(f"**Task Achievement:** {result['Task Achievement']}")
            st.write(f"**Coherence:** {result['Coherence']}")
            st.write(f"**Lexical Resource:** {result['Lexical Resource']}")
            st.write(f"**Grammar:** {result['Grammar']}")
            st.write(f"**Overall Band Score:** {result['Overall Band Score']}")

            # Display detailed feedback
            st.subheader("Feedback & Improvements:")
            st.write(result["feedback"])  # Full feedback from GPT
    
    else:
        st.warning("No valid response provided. Please type your response or upload a file.")

if __name__ == "__main__":
    # Start your app (Streamlit or another framework)
    st.write("App is running! Use `streamlit run your_script.py`")
