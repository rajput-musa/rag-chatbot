# RAG Chatbot

This is a live, interactive RAG (Retrieval-Augmented Generation) system designed to answer questions from a corpus of documents with high accuracy and speed.

## Live Demo
[Click here to access the live application](https://huggingface.co/spaces/MusaR/rag-chatbot)

## Architecture
This project implements an advanced RAG pipeline leveraging third-party services for optimal performance:
- **Vector Database:** Pinecone (for fast, scalable vector search)
- **LLM (Generator):** Groq with Llama 3 70B (for lightning-fast, high-quality generation)
- **Re-ranker:** Cohere Rerank API (to increase the precision of retrieved context)
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces

## Key Features
- **Advanced Retrieval:** The system first retrieves a large set of candidate documents from Pinecone and then uses Cohere's powerful re-ranker to select the most relevant context.
- **Structured Output:** The LLM is prompted to return answers in a structured JSON format (summary, key points, confidence score) for reliable and clean presentation in the UI.
- **Verifiability:** The app displays the exact re-ranked source passages used to generate the answer, ensuring transparency and trust.
