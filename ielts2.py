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

# In[4]:

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Ensure spaCy model is installed
#def ensure_spacy_model():
#    try:
#        spacy.load("en_core_web_sm")
#    except OSError:
#        print("Downloading spaCy model...")
#        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
#        spacy.load("en_core_web_sm")  # Reload after downloading

#ensure_spacy_model()
    

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


# Streamlit UI
st.title("IELTS Writing Test")

if st.button("Generate IELTS Question"):
    ielts_question = generate_ielts_question()
    st.session_state['ielts_question'] = ielts_question

if 'ielts_question' in st.session_state:
    st.subheader("Your IELTS Question:")
    st.write(st.session_state['ielts_question'])
    
    # Input options
    user_input = st.text_area("Type your response:")
    uploaded_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"])
    
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
    import streamlit as st
    st.write("App is running! Use `streamlit run your_script.py`")
