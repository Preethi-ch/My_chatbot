import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.markdown("""
<style>
    .stApp {
        background: #121212;  /* Dark background */
        background-image: radial-gradient(#1e1e1e 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #cddc39;  /* Soft lime green */
    }
    .user-msg {
        background: #1e1e1e !important;  /* Dark gray */
        border-radius: 15px !important;
        border: 2px solid #8bc34a !important;  /* Green border */
        color: #ffffff !important;
    }
    .bot-msg {
        background: #263238 !important;  /* Dark cyan-gray */
        border-radius: 15px !important;
        border: 2px solid #03a9f4 !important;  /* Neon blue border */
        color: #81d4fa !important;  /* Light blue text */
    }
    .stChatInput {
        background: #1e1e1e;
        color: #ffffff;
        border: 2px solid #8bc34a !important;
    }
</style>
""", unsafe_allow_html=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2') 

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

st.markdown('<h1 class="chat-font">🤖 Preethi Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as Preethi!</h3>', unsafe_allow_html=True)
st.markdown("---")

def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']  # Return the closest answer
    return None

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Find the closest answer
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                # Limit the response to 2-3 lines
                short_answer = " ".join(retrieved_answer.split()[:30]) + "..."
                response = f"**Preethi**:\n{short_answer}"
            else:
                response = "**Preethi**:\nI'm sorry, I cannot answer that question."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
