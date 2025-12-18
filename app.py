import os
import streamlit as st
import pickle
import pandas as pd
import requests
from typing import List, Dict

# Import your RAG backend
from rag_backend import rag_reply
# -----------------------------
# Helper functions
# -----------------------------
def poster_fetch(movie_id: int) -> str:
    """Fetch poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e00e6b991e8c3257fa29d03c158f6b2a&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get("poster_path")
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return ""

def recommend(movie: str):
    """Return top 6 recommended movies and posters."""
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:7]
    recommended_movies = []
    recommended_posters = []
    for i in distances:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(poster_fetch(movie_id))
    return recommended_movies, recommended_posters


movie_list = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movie_list)
similarity = pickle.load(open("similarity.pkl", "rb"))


st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")
st.title("🎬 MovieMate: Recommendation & Chat")

# Tabs for Recommendations and Chat
tab_rec, tab_chat = st.tabs(["Recommendations", "Chatbot"])

with tab_rec:
    selected_movie = st.selectbox("What movie are you looking for?", movies['title'].values)
    if st.button("Recommend"):
        names, posters = recommend(selected_movie)
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(names):
                col_width = 200  # or calculate based on screen layout
                col.image(posters[idx], caption=names[idx], width=col_width) 


with tab_chat:
    st.subheader("Chat with MovieMate")
    st.caption("Powered by Azure OpenAI. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in your environment.")

    # Chat CSS styling
    st.markdown(
        """
        <style>
        .chat-wrap { display: flex; flex-direction: column; gap: 0.5rem; }
        .chat-msg { padding: 0.6rem 0.8rem; border-radius: 12px; max-width: 72ch; line-height: 1.5; }
        .chat-user { background: #e6f2ff; color: #0b3d91; align-self: flex-end; text-align: right; border: 1px solid #cfe2ff; }
        .chat-bot  { background: #e9f7ef; color: #145a32; align-self: flex-start; text-align: left; border: 1px solid #d4efdf; }
        .chat-sep  { height: 0; border-top: 1px solid #eee; margin: 0.25rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [] 
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""

    # Input for user message
    col1, col2 = st.columns([3, 1])
    with col1:
        user_q = st.text_input(
            "Your message",
            value=st.session_state["pending_query"],
            placeholder="Ask about movies, actors, directors, or get recommendations..."
        )
    with col2:
        send = st.button("Send")

    # Handle sending message
    if send and user_q.strip():
        st.session_state["messages"].append({"role": "user", "content": user_q.strip()})
        try:
            if rag_reply is None:
                raise RuntimeError("Chat backend not available. Ensure rag_backend.py exists and is configured.")
            reply_text, updated_history = rag_reply(
                user_q.strip(), 
                hf_token=None,
            history=st.session_state["messages"])
            st.session_state["messages"] = updated_history
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})

        st.session_state["pending_query"] = ""
    else:
        st.session_state["pending_query"] = user_q

    # Display chat messages
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state["messages"]):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        cls = "chat-user" if role == "user" else "chat-bot"
        st.markdown(f'<div class="chat-msg {cls}">{content}</div>', unsafe_allow_html=True)
        if i < len(st.session_state["messages"]) - 1:
            st.markdown('<div class="chat-sep"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")