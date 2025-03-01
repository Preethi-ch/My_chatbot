import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

st.markdown("""
<style>
    .stApp {
        background: #121212;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #cddc39;
    }
    .user-msg {
        background: #1e1e1e !important;
        border-radius: 15px !important;
        border: 2px solid #8bc34a !important;
        color: #ffffff !important;
    }
    .bot-msg {
        background: #263238 !important;
        border-radius: 15px !important;
        border: 2px solid #03a9f4 !important;
        color: #81d4fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini API
genai.configure(api_key="AIzaSyADZJ11fXuCbq6lLrTmu02zEcdx0DYja2Q")
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Sentence Transformer for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Dataset must have 'question' and 'answer' columns
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("CSV must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", axis=1)
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df, faiss_index = load_data()

st.markdown('<h1 class="chat-font">🤖 Preethi Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<h3 class='chat-font'>Ask me anything, and I'll respond as Preethi!</h3>", unsafe_allow_html=True)

st.markdown("---")

# Function to find the closest question
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']
    return None

# Function to generate refined answers using AI
def generate_refined_answer(query, retrieved_answer, chat_history):
    chat_context = "\n".join([f"User: {msg['content']}" for msg in chat_history if msg["role"] == "user"][-5:])  # Last 5 user inputs
    prompt = f"""
    You are Preethi, a 3rd-year BTech AIML student. Answer in a friendly, conversational tone.
    Previous Chat History:
    {chat_context}
    Current Question: {query}
    Retrieved Answer: {retrieved_answer}
    
    - Ensure the answer is at least 3 lines long.
    - Use correct grammar and make it engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text if response else retrieved_answer

# Function to calculate age dynamically
def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%d %B %Y")
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return f"I am {age} years old. Time flies, right? Feels like I just started my journey!"
    except Exception:
        return "I couldn't calculate my age, but I'm forever young at heart!"

# Store conversation history **without clearing after 5 prompts**
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            if "age" in prompt.lower():
                response = calculate_age("13 October 2004")  # Hardcoded DOB
            elif "your name" in prompt.lower() and "jyo" in prompt.lower():
                response = "My name is Preethi, not Jyo."
            else:
                retrieved_answer = find_closest_question(prompt, faiss_index, df)
                response = generate_refined_answer(prompt, retrieved_answer, st.session_state.messages) if retrieved_answer else "I don't know that yet, but I'm learning!"
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
