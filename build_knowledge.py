from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os
import pickle


# ==============================
# 1. READ ALL PDFs
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
                text = text.replace("\n"," ")
                text = " ".join(text.split())
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
# 4. CREATE EMBEDDINGS
# ==============================

print("\nCreating embeddings...")

embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embeddings created!")


# ==============================
# 5. SAVE KNOWLEDGE BASE
# ==============================

knowledge_base = {
    "chunks": chunks,
    "embeddings": embeddings
}

with open("knowledge_base.pkl", "wb") as file:

    pickle.dump(knowledge_base, file)


print("\nKnowledge base saved successfully! 🚀")