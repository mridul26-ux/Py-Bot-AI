import streamlit as st
import nltk
import random
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- 1. SETUP & TRAINING (Runs once when app starts) ---
@st.cache_resource
def train_model():
    # Downloads needed for the local machine or server
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Your Final Training Data
    training_data = [
        ("hi", "greeting"), ("hello", "greeting"), ("hey", "greeting"), ("hii", "greeting"), ("greetings", "greeting"),
        ("how are you", "status"), ("how is it going", "status"),("how is everything going on", "status"),("what is the status ", "status"),
        ("what is your name", "identity"), ("who are you", "identity"),("your name", "identity"), ("tell me your name", "identity"),
        ("is it raining", "weather"), ("what is the weather", "weather"), 
        ("tell me about weather", "weather"), ("temperature", "weather"),
        ("bye", "goodbye"), ("goodbye", "goodbye"), ("exit", "goodbye")
    ]
    X_train, y_train = zip(*training_data)
    
    # Your ML Pipeline (Tfidf + Naive Bayes)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(X_train, y_train)
    return model

model = train_model()

# Your Response Pools
responses = {
    "greeting": ["Hello! How can I help you today?", "Hi there!", "Hey! What's up?"],
    "status": ["I'm doing great! Just processing some data.", "I'm a bot, so I'm always 100%!"],
    "identity": ["I'm PyBot, your NLTK-powered assistant.", "You can call me PyBot."],
    "weather": ["I can't see outside, but you can refer to https://www.accuweather.com/ for live updates."],
    "goodbye": ["Goodbye! Have a great day.", "See you later!", "Bye! 👋"],
    "unknown": ["I'm not sure I understand. You can also refer to https://www.google.com/ for more info."]
}

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="PyBot AI", page_icon="🤖")
st.title("🤖 PyBot: AI Edition")
st.markdown("Ask me about the weather, my status, or just say hello!")

# Initialize chat history (This keeps the conversation visible)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. CHAT LOGIC ---
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pre-process the input
    tokens = word_tokenize(prompt.lower())
    clean_input = " ".join([w for w in tokens if w.isalnum()])
    
    if not clean_input:
        full_response = "I'm listening... feel free to type something!"
    else:
        # Confidence Check (The logic you perfected)
        probs = model.predict_proba([clean_input])[0]
        max_prob = max(probs)
        max_idx = list(probs).index(max_prob)
        intent = model.classes_[max_idx]

        # Your 0.3 threshold
        if max_prob < 0.15:
            intent = "unknown"
            
        full_response = random.choice(responses.get(intent, responses["unknown"]))

    # Display Bot response and add to history
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})