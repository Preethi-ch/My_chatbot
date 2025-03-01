import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Streamlit page styling
st.markdown("""
<style>
    .stApp {
        background: #121212;  
        background-image: radial-gradient(#1e1e1e 1px, transparent 1px);
        background-size: 20px 20px;
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

# Configure Gemini API (Replace with your actual API key)
genai.configure(api_key="AIzaSyADZJ11fXuCbq6lLrTmu02zEcdx0DYja2Q")
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with actual dataset
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()

        df['context'] = df.apply(lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", axis=1)
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

st.markdown('<h1 class="chat-font">🤖 Preethi Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as Preethi!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find closest question
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  

    if I.size > 0 and I[0][0] < len(df):  
        retrieved_answer = df.iloc[I[0][0]]['answer']
        if retrieved_answer.lower() in ["preethi...", "i don't know", ""]:
            return None  # Ignore irrelevant responses
        return retrieved_answer

    return None  # No close match found

# Function to generate AI-based response (only if dataset doesn't have a good answer)
def generate_ai_response(query):
    prompt = f"""Answer the following question in 2-3 lines in a friendly tone:
    Question: {query}
    Keep the response natural and engaging."""
    
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip() if response.text else "I'm not sure about that."
    except:
        return "Sorry, I couldn't generate a response right now."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            response = retrieved_answer if retrieved_answer else generate_ai_response(prompt)
            response = f"**Preethi:** {response}"  # Formatting response
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
