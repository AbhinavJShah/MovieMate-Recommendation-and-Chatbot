# 🎬 MovieMate – AI Movie Recommendation & Chatbot

MovieMate is an AI-powered movie recommendation system combined with a conversational chatbot.  
It uses **content-based filtering** for recommendations and an **Azure OpenAI–powered chatbot** to answer movie-related questions interactively.

---

## 🚀 Features

- 🎥 Content-based movie recommendation using cosine similarity
- 🧠 AI chatbot for movie-related questions (actors, genres, directors, suggestions)
- 💬 Conversational UI with chat history
- 🖼️ Movie posters fetched dynamically from TMDB API
- 🌐 Interactive Streamlit web interface
- ☁️ Azure OpenAI integration for chat responses

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas & NumPy**
- **Scikit-learn** (similarity computation)
- **Azure OpenAI (Chat Completions)**
- **TMDB API**
- **Pickle** (model persistence)

---

## 📂 Project Structure
├── app.py # Streamlit frontend
├── rag_backend.py # Azure OpenAI chatbot backend
├── Movie Recommendation.ipynb # Model training & experimentation
├── movie_dict.pkl # Movie metadata
├── similarity.pkl # Precomputed similarity matrix
├── requirements.txt
└── README.md
