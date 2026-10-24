import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import pickle
import time


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DataMentor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: #0e1117;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* Main title */

.datamentor-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.datamentor-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 17px;
    margin-bottom: 30px;
}


/* Welcome card */

.welcome-card {
    padding: 25px;
    border-radius: 18px;
    background: #171b24;
    border: 1px solid #292f3a;
    margin-bottom: 25px;
}

.welcome-title {
    font-size: 23px;
    font-weight: 600;
    margin-bottom: 8px;
}

.welcome-text {
    color: #aeb4c0;
    line-height: 1.6;
}


/* Sidebar */

.sidebar-title {
    font-size: 25px;
    font-weight: 700;
}

.sidebar-text {
    color: #9ca3af;
    line-height: 1.5;
}


/* Status */

.status-box {
    padding: 12px;
    border-radius: 12px;
    background: #151a22;
    border: 1px solid #292f3a;
    margin-top: 15px;
    color: #b9c0cc;
}


/* Chat */

[data-testid="stChatMessage"] {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 DataMentor")
    st.caption("Your AI learning companion")

    st.divider()

    st.markdown("### 💬 Recent Chats")

    if "recent_chats" not in st.session_state:
        st.session_state.recent_chats = []

    if st.session_state.recent_chats:
        for chat in st.session_state.recent_chats[:8]:
            st.markdown(f"• {chat}")
    else:
        st.caption("No recent chats yet.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recent_chats = []
        st.rerun()

# =========================================================
# HEADER
# =========================================================

st.markdown("""<div style="
text-align:center;
padding:20px 0 10px 0;
">
<div style="
font-size:42px;
font-weight:700;
">
🤖 DataMentor
</div>
<div style="
font-size:17px;
color:#9ca3af;
margin-top:6px;
">
Your intelligent educational AI tutor
</div>
</div>""", unsafe_allow_html=True)
# =========================================================
# LOAD KNOWLEDGE BASE + MODELS
# =========================================================

@st.cache_resource
def load_datamentor():

    # Load knowledge base

    with open("knowledge_base.pkl", "rb") as file:
        knowledge_base = pickle.load(file)

    chunks = knowledge_base["chunks"]
    chunk_embeddings = knowledge_base["embeddings"]

    # Load embedding model locally

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        local_files_only=True
    )

    # Connect to Gemini

    client = genai.Client()

    return chunks, chunk_embeddings, model, client


# =========================================================
# INITIALIZE
# =========================================================

try:

    chunks, chunk_embeddings, model, client = load_datamentor()

except Exception as e:

    st.error("Unable to initialize DataMentor.")
    st.code(str(e))
    st.stop()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# WELCOME SCREEN
# =========================================================

if len(st.session_state.messages) == 0:

    st.markdown("""<div class="welcome-card">
    <div class="welcome-title">
    👋 Welcome to DataMentor
    </div>
    <div class="welcome-text">
    I'm your AI learning companion. Ask me questions about
    your study material or explore concepts beyond it.
    </div>
    </div>""", unsafe_allow_html=True)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask DataMentor anything..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # Save user message

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    if "recent_chats" not in st.session_state:
        st.session_state.recent_chats = []

    st.session_state.recent_chats.insert(0, question)

# Keep only the latest 8 chats
    st.session_state.recent_chats = st.session_state.recent_chats[:8]

    # Display user message

    with st.chat_message("user"):

        st.markdown(question)


    # =====================================================
    # EMBED QUESTION
    # =====================================================
    start=time.time()
    question_embedding = model.encode([question])
    print("Embedding time:", time.time()-start)

    # =====================================================
    # FIND RELEVANT PDF CONTENT
    # =====================================================
    start=time.time()
    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]
    print("Search  time:",time.time()-start)

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    threshold = 0.35


    if best_score >= threshold:

        # Get top 3 relevant chunks

        top_indices = similarities.argsort()[-3:][::-1]

        relevant_chunks = []

        for index in top_indices:

            relevant_chunks.append(chunks[index])

        context = "\n\n".join(relevant_chunks)

        clean_context = context.replace("\n", " ")

    else:

        clean_context = ""


    # =====================================================
    # CREATE PROMPT
    # =====================================================

    prompt = f"""
You are DataMentor, a friendly and intelligent educational AI tutor.

Answer the user's question naturally and helpfully.

You have access to study material from the DataMentor knowledge base.
Use that material when it is relevant, but you are NOT restricted to it.

Rules:

1. If reference material is provided and relevant, use it to improve
   your answer.

2. You may use your general knowledge for questions that go beyond
   the reference material.

3. Do not pretend that general knowledge came from the knowledge base.

4. Explain concepts clearly at a student-friendly level.

5. Give examples when they help understanding.

6. Do not unnecessarily copy the wording or structure of the PDF.

7. If no relevant reference material is provided, simply answer the
   question using your general knowledge.

8. Do not invent facts.

REFERENCE MATERIAL:
{clean_context}

USER QUESTION:
{question}

Answer naturally and directly.
"""


    # =====================================================
    # ASK GEMINI
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner("DataMentor is thinking... 🤔"):

            try:
                start=time.time()

                interaction = client.interactions.create(
                    model="gemini-3.5-flash-lite",
                    input=prompt
                )
                print("Gemini time:",time.time()-start)

                answer = interaction.output_text

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    answer="""
                    
⚠️ **DataMentor is temporarily busy.**

You've reached the Gemini free-tier request limit.
Please wait a little while and try again.
"""
                else:
                    answer = f"Sorry, I couldn't generate an answer.\n\nError: {e}"


               

        st.markdown(answer)


    # =====================================================
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })