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
#NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
#os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Set the custom directory for NLTK data
#nltk.data.path.append(NLTK_DATA_PATH)
#os.environ['NLTK_DATA'] = NLTK_DATA_PATH  # Manually set env variable

# Download required NLTK corpora
#nltk.download('punkt', download_dir=NLTK_DATA_PATH)  # Tokenization
#nltk.download('wordnet', download_dir=NLTK_DATA_PATH)  # Lemmatization
#nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)  # POS Tagging
#nltk.download('brown', download_dir=NLTK_DATA_PATH)  # Required for TextBlob

# Force load the corpora before using TextBlob
#try:
#    Word("hello").synsets  # Forces TextBlob to check its corpora
#    _ = Spelling()  # Ensures spelling corpus is loaded
#except Exception as e:
#    st.error(f"TextBlob corpus loading error: {str(e)}")
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

IELTS_BAND_DESCRIPTORS = """
### IELTS Writing Band Descriptors (Task 2) - Official Criteria

#### **Task Achievement**
- **Band 9**: Fully addresses all parts of the task, presents a well-developed position with fully extended and well-supported ideas.
- **Band 8**: Covers all task requirements sufficiently, presents a clear and well-developed position.
- **Band 7**: Addresses all parts of the task, though some ideas may lack full support or development.
- **Band 6**: Addresses the task, but some points may be underdeveloped or missing.
- **Band 5**: Generally addresses the task but lacks sufficient development.
- **Band 4 or below**: Does not fully address the task or is off-topic.

#### **Coherence & Cohesion**
- **Band 9**: Logical sequencing, seamless flow, and skillful paragraphing.
- **Band 8**: Well-organized response with logical progression and good paragraphing.
- **Band 7**: Clear progression, appropriate paragraphing but occasional lapses.
- **Band 6**: Ideas are generally arranged logically but may have some cohesion issues.
- **Band 5**: Lacks coherence; linking devices may be inaccurate or overused.
- **Band 4 or below**: Disjointed ideas with weak organization.

#### **Lexical Resource**
- **Band 9**: Wide, precise, and sophisticated vocabulary, with natural collocations.
- **Band 8**: A wide range of vocabulary with minor occasional inaccuracies.
- **Band 7**: Uses a good range of vocabulary but with occasional repetition or inappropriate word choice.
- **Band 6**: Vocabulary is adequate but lacks flexibility.
- **Band 5**: Limited vocabulary, frequent repetition, and basic expressions.
- **Band 4 or below**: Very basic vocabulary with frequent errors that impede communication.

#### **Grammatical Range & Accuracy**
- **Band 9**: A wide range of structures, complex sentences are used naturally.
- **Band 8**: A good mix of complex and simple structures, minor errors may occur.
- **Band 7**: Uses a variety of sentence forms, but some errors remain.
- **Band 6**: Uses some complex structures but errors may persist.
- **Band 5**: Limited range of structures, frequent grammar mistakes.
- **Band 4 or below**: Frequent errors affecting communication.

Use this rubric to score an IELTS writing response based on the essay below.
"""



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

    # Cohesion markers (transition words)
    cohesion_markers = [token.text.lower() for token in doc if token.dep_ in ["mark", "advmod"]]
    cohesion_score = len(cohesion_markers) / sentence_count if sentence_count > 0 else 0

    # Lexical repetition (overused words)
    common_words = Counter([token.text.lower() for token in doc if token.is_alpha]).most_common(5)

    # Sentence complexity (complex sentence ratio)
    complex_sentences = sum(1 for token in doc if token.dep_ in ["csubj", "xcomp", "advcl"])
    complexity_score = complex_sentences / sentence_count if sentence_count > 0 else 0

    # GPT-4 Evaluation with Band Descriptors
    messages = [
    {"role": "system", "content": f"You are an IELTS examiner. Use the following IELTS Band Descriptors for grading:\n\n{IELTS_BAND_DESCRIPTORS}"},
    {"role": "user", "content": f"""Prompt: {prompt}\n\nEssay: {text}\n\n
Based on the IELTS criteria, provide the following response with this strict format:
---
Task Achievement: [Score: Band X.X]
Coherence: [Score: Band X.X]
Lexical Resource: [Score: Band X.X]
Grammar: [Score: Band X.X]
Overall Band Score: [Score: Band X.X]
---
Justification:
[List reasons for each score]
---
Suggestions for Improved Score"
[List suggestions to improve their score]
Your output MUST follow this exact format. Do NOT change the structure. Do NOT add extra sections. Only use the format provided."""}
]
    
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::B8gProGu",
        messages=messages
    )

    feedback_text = response.choices[0].message.content

    # Extracting scores using regex
    scores = {
        "Task Achievement": None,
        "Coherence": None,
        "Lexical Resource": None,
        "Grammar": None,
        "Overall Band Score": None
    }

    patterns = {
        "Task Achievement": r"Task Achievement\s*\nScore:\s*Band (\d(?:\.\d)?)",
        "Coherence": r"Coherence\s*\nScore:\s*Band (\d(?:\.\d)?)",
        "Lexical Resource": r"Lexical Resource\s*\(Vocabulary\)\s*\nScore:\s*Band (\d(?:\.\d)?)",
        "Grammar": r"Grammatical Range & Accuracy\s*\nScore:\s*Band (\d(?:\.\d)?)",
        "Overall Band Score": r"Overall Band Score\s*\nScore:\s*Band (\d(?:\.\d)?)"
    }

    scores = {key: "Not Provided" for key in patterns}  # Default if extraction fails
    for category, pattern in patterns.items():
        match = re.search(pattern, feedback_text, re.IGNORECASE)
        if match:
            scores[category] = float(match.group(1))  # Convert to float

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": round(lexical_diversity, 2),
        "cohesion_score": round(cohesion_score, 2),
        "common_words": common_words,
        "complex_sentence_ratio": round(complexity_score, 2),
        "Task Achievement": scores["Task Achievement"],
        "Coherence": scores["Coherence"],
        "Lexical Resource": scores["Lexical Resource"],
        "Grammar": scores["Grammar"],
        "Overall Band Score": scores["Overall Band Score"],
        "feedback": feedback_text
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
    # Clear previous input fields
    if 'user_input' in st.session_state:
        del st.session_state['user_input']  # Correct way to clear text box
    
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file'] 

if 'ielts_question' in st.session_state:
    st.subheader("Your IELTS Question:")
    st.write(st.session_state['ielts_question'])
    
    # Input options
    user_input = st.text_area("Type your response:", height=300, key="user_input")
    uploaded_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"], key="uploaded_file")

if st.button("Submit Response"):
    essay_text = user_input.strip() if user_input else None
    if uploaded_file:
        essay_text = extract_text_from_file(uploaded_file)

    if essay_text:
        with st.spinner("Analyzing your response... Please wait."):
            feedback = analyze_essay(essay_text, st.session_state['ielts_question'])

        # Display analysis results
        st.subheader("Assessment Result:")
        st.write(f"**Word Count:** {feedback['word_count']}")
        st.write(f"**Sentence Count:** {feedback['sentence_count']}")
        st.write(f"**Lexical Diversity:** {feedback['lexical_diversity']:.2f}")
        st.write(f"**Cohesion Score:** {feedback['cohesion_score']:.2f}")
        st.write(f"**Complex Sentence Ratio:** {feedback['complex_sentence_ratio']:.2f}")

        # Format and display common words as a table
        import pandas as pd
        common_words_df = pd.DataFrame(feedback["common_words"], columns=["Word", "Count"])
        st.write("### Most Common Words")
        st.table(common_words_df)

        # Display band scores
        st.subheader("Band Scores:")
        st.write(f"**Task Achievement:** {feedback['Task Achievement']}")
        st.write(f"**Coherence:** {feedback['Coherence']}")
        st.write(f"**Lexical Resource:** {feedback['Lexical Resource']}")
        st.write(f"**Grammar:** {feedback['Grammar']}")
        st.write(f"**Overall Band Score:** {feedback['Overall Band Score']}")

        # Display detailed feedback
        st.subheader("Feedback & Improvements:")
        st.write(feedback["feedback"])  # Full GPT-4 generated feedback

    else:
        st.warning("No valid response provided. Please type your response or upload a file.")


if __name__ == "__main__":
    # Start your app (Streamlit or another framework)
    st.write("App is running! Use `streamlit run your_script.py`")
