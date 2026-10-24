# 🤖 DataMentor

### Your Intelligent Educational AI Tutor

DataMentor is an AI-powered educational chatbot built using Python, Streamlit, Retrieval-Augmented Generation (RAG), Sentence Transformers, and Google's Gemini API.

It helps students understand academic concepts by combining information retrieved from study materials with the general knowledge capabilities of a Large Language Model (LLM).

## 🚀 Features

* AI-powered educational question answering.
* Semantic search using Sentence Transformers.
* Retrieval-Augmented Generation (RAG).
* PDF-based knowledge retrieval.
* General question answering beyond uploaded study materials.
* Interactive chatbot interface using Streamlit.
* Cosine similarity for retrieving relevant information.

## 🛠️ Technology Stack

| Technology            | Purpose                          |
| --------------------- | -------------------------------- |
| Python                | Core programming language        |
| Streamlit             | Web interface                    |
| Gemini API            | AI-generated responses           |
| Sentence Transformers | Text embeddings                  |
| Scikit-learn          | Cosine similarity                |
| Pickle                | Local knowledge-base storage     |
| RAG                   | Context-aware question answering |

## 🧠 How DataMentor Works

1. Study materials are processed into smaller text chunks.
2. Sentence Transformers generate embeddings for those chunks.
3. The embeddings are stored in a local knowledge base.
4. A user submits a question through Streamlit.
5. DataMentor performs semantic similarity search.
6. Relevant context is retrieved from the knowledge base.
7. Gemini generates a student-friendly answer using relevant context and, when appropriate, general knowledge.

## 💻 Running the Project Locally

Clone the repository and open the project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key locally using the `GEMINI_API_KEY` environment variable.

Provide your own appropriately licensed study materials and build the local knowledge base using:

```bash
python build_knowledge.py
```

Start the application:

```bash
streamlit run web_app.py
```

## 📌 Project Purpose

DataMentor demonstrates how Retrieval-Augmented Generation can enhance an educational chatbot by combining semantic retrieval with generative AI.

The project focuses on making learning more interactive, accessible, and student-friendly.

## 🔐 Security Note

API credentials, local study materials, and generated knowledge-base files are excluded from the public repository.
