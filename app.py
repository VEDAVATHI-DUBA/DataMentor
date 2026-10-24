'''from pypdf import PdfReader
import os

folder = "Knowledge"

all_text = ""

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(folder, filename)

        print("Reading:", filename)

        reader = PdfReader(filepath)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                all_text += text + "\n"

print("\nTotal characters extracted:", len(all_text))
print("\nFirst 5000 characters:\n")
print(all_text[:5000])'''
'''from pypdf import PdfReader
import os

folder = "Knowledge"

all_text = ""

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(folder, filename)

        print("Reading:", filename)

        reader = PdfReader(filepath)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                all_text += text + "\n"

        print("Finished:", filename)
        chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(all_text):
    end = start + chunk_size
    chunk = all_text[start:end]

    chunks.append(chunk)

    start = end - overlap

print("\nTotal chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0])'''


'''from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


# -----------------------------------------
# 1. READ ALL PDFS
# -----------------------------------------

folder = "knowledge"

all_text = ""

for filename in os.listdir(folder):

    if filename.endswith(".pdf"):

        filepath = os.path.join(folder, filename)

        print("Reading:", filename)

        reader = PdfReader(filepath)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                all_text += text + "\n"

        print("Finished:", filename)


# -----------------------------------------
# 2. CREATE CHUNKS
# -----------------------------------------

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(all_text):

    end = start + chunk_size

    chunk = all_text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("\nTotal chunks:", len(chunks))


# -----------------------------------------
# 3. LOAD EMBEDDING MODEL
# -----------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# -----------------------------------------
# 4. CREATE EMBEDDINGS FOR CHUNKS
# -----------------------------------------

print("\nCreating embeddings...")

chunk_embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embeddings created!")


# -----------------------------------------
# 5. ASK A QUESTION
# -----------------------------------------

question = input("\nAsk DataMentor: ")


# -----------------------------------------
# 6. EMBEDDING OF QUESTION
# -----------------------------------------

question_embedding = model.encode([question])


# -----------------------------------------
# 7. CALCULATE SIMILARITY
# -----------------------------------------

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]


# -----------------------------------------
# 8. GET TOP 3 RESULTS
# -----------------------------------------

top_indices = similarities.argsort()[-3:][::-1]


print("\nRelevant information:\n")

for index in top_indices:

    print("--------------------------------")
    print(chunks[index])
    print("Similarity:", similarities[index])'''
'''from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import os


# ==============================
# 1. READ ALL PDFS
# ==============================

folder = "knowledge"
all_text = ""

for filename in os.listdir(folder):

    if filename.endswith(".pdf"):

        filepath = os.path.join(folder, filename)

        print("Reading:", filename)

        reader = PdfReader(filepath)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                all_text += text + "\n"

        print("Finished:", filename)


# ==============================
# 2. CREATE CHUNKS
# ==============================

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(all_text):

    end = start + chunk_size

    chunk = all_text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("\nTotal chunks:", len(chunks))


# ==============================
# 3. LOAD EMBEDDING MODEL
# ==============================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==============================
# 4. CREATE CHUNK EMBEDDINGS
# ==============================

print("\nCreating embeddings...")

chunk_embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embeddings created!")


# ==============================
# 5. CONNECT TO GEMINI
# ==============================

client = genai.Client()

print("\nDataMentor is ready! 🚀")


# ==============================
# 6. ASK QUESTION
# ==============================

question = input("\nAsk DataMentor: ")


# ==============================
# 7. EMBED QUESTION
# ==============================

question_embedding = model.encode([question])


# ==============================
# 8. FIND RELEVANT CHUNKS
# ==============================

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

top_indices = similarities.argsort()[-3:][::-1]

relevant_chunks = []

for index in top_indices:

    relevant_chunks.append(chunks[index])


context = "\n\n".join(relevant_chunks)


# ==============================
# 9. SEND QUESTION + CONTEXT
# ==============================

prompt = f"""
You are DataMentor, an educational chatbot.

Answer the user's question using the provided material.

If the provided material does not contain enough information,
say that the information is not available in the provided
knowledge base.

Do not invent information.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

Give a clear and easy-to-understand answer.
"""

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)


# ==============================
# 10. DISPLAY ANSWER
# ==============================

print("\n🤖 DataMentor:")
print(interaction.output_text)'''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import pickle


# ==============================
# 1. LOAD SAVED KNOWLEDGE BASE
# ==============================

print("Loading knowledge base...")

with open("knowledge_base.pkl", "rb") as file:
    knowledge_base = pickle.load(file)

chunks = knowledge_base["chunks"]
chunk_embeddings = knowledge_base["embeddings"]

print("Knowledge base loaded!")


# ==============================
# 2. LOAD EMBEDDING MODEL
# ==============================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

print("Embedding model loaded!")


# ==============================
# 3. CONNECT TO GEMINI
# ==============================

client = genai.Client()

print("\n🤖 DataMentor is ready!")


# ==============================
# 4. ASK QUESTION
# ==============================

question = input("\nAsk DataMentor: ")


# ==============================
# 5. EMBED QUESTION
# ==============================

question_embedding = model.encode([question])


# ==============================
# 6. FIND RELEVANT CHUNKS
# ==============================

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

# Get the best matching chunk
best_index = similarities.argmax()
best_score = similarities[best_index]

# Minimum similarity required to consider PDF content relevant
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

    # No sufficiently relevant PDF content
    clean_context = ""


# ==============================
# 7. CREATE PROMPT
# ==============================


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
4. Explain concepts clearly and at a student-friendly level.
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


# ==============================
# 8. ASK GEMINI
# ==============================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)


# ==============================
# 9. DISPLAY ANSWER
# ==============================

print("\n🤖 DataMentor:")
print(interaction.output_text)