# Advanced RAG Document Chatbot

This project is an interactive RAG (Retrieval-Augmented Generation) chatbot designed to answer questions from a corpus of documents. It leverages several modern AI services and techniques to provide accurate, sourced answers.

## Live Demo
[Click here to access the live application](https://huggingface.co/spaces/MusaR/rag-chatbot) 
*(Note: The application is demonstrated using the textbook "Operating Systems Internals and Design Principles" by William Stallings.)*

## Core Architecture & Technologies
This chatbot implements a multi-stage RAG pipeline:

*   **Vector Database:** Pinecone (for efficient semantic search and storage of document embeddings)
*   **LLM (Generator):** Groq API with Llama 3 70B (for fast and high-quality answer generation)
*   **Re-ranker:** Cohere Rerank API (to refine and prioritize retrieved context for accuracy)
*   **Embedding Model:** `all-MiniLM-L6-v2` (for generating text embeddings)
*   **Frontend:** Streamlit (for the interactive web interface)
*   **Deployment:** Hugging Face Spaces (utilizing a custom Docker environment)

## Key Features
*   **Advanced Retrieval Pipeline:** Implemented a retrieve-then-rerank strategy to enhance the quality of context provided to the LLM.
*   **Sourced Answers:** The system is designed to provide answers grounded in the provided documents, with a mechanism to display the source passages.


## How to Use with Your Own Documents
This deployed version is configured to use a pre-existing Pinecone index. To use this system with your own documents:
1.  Ensure you have API keys for Pinecone, Groq, and Cohere.
2.  You will need to create and populate your own Pinecone index (dimension: 384, metric: cosine).
3.  A [sample ingestion script/notebook (link to your Gist or simplified Colab notebook here)] can be used as a starting point to process your PDFs and upload them to your Pinecone index.
4.  This application can then be forked on Hugging Face Spaces, and your API keys can be added to the secrets of your forked Space to connect to your data.

