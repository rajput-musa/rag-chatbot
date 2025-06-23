# 🔍 RAG Document Chatbot

An interactive **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on any document corpus. It combines semantic search, re-ranking, and large language models (LLMs) to deliver accurate, context-aware, and **source-backed answers**.

## 🚀 Live Demo

▶️ [Try it on Hugging Face Spaces](https://huggingface.co/spaces/MusaR/rag-chatbot)

> *(Demo uses the textbook: "Operating Systems Internals and Design Principles" by William Stallings.)*

---

## ⚙️ How It Works

The system follows a multi-step RAG pipeline:

1. User submits a question.
2. The question is embedded using `all-MiniLM-L6-v2`.
3. Pinecone retrieves semantically relevant document chunks.
4. Cohere's Rerank API refines the context ordering.
5. Groq's Llama 3 70B generates a final answer using the top-ranked context.
6. The chatbot displays the answer along with the referenced sources.

---

## 🧠 Core Technologies

| Component  | Technology / Service                                            |
| ---------- | --------------------------------------------------------------- |
| Vector DB  | Pinecone                                                        |
| LLM        | Llama 3 70B via [Groq API](https://console.groq.com/)           |
| Re-ranking | [Cohere Rerank](https://cohere.com/rerank)                      |
| Embeddings | `all-MiniLM-L6-v2` from `sentence-transformers`                 |
| Frontend   | [Streamlit](https://streamlit.io/)                              |
| Deployment | [Hugging Face Spaces](https://huggingface.co/spaces) via Docker |

---

## ✨ Features

* 🔍 **Semantic Search + Reranking** for high-quality context retrieval.
* 📚 **Source-aware Answers** with citations from documents.
* ⚡ **Low-latency Generation** using Groq's LLM API.
* 🌐 **Web-based UI** built with Streamlit.
* 📦 **Dockerized** for easy deployment and portability.

---

## 📁 Using Your Own Documents

To customize the chatbot with your own data:

### 🔑 1. Get API Keys

Create free accounts and obtain API keys from:

* Pinecone: [https://www.pinecone.io/](https://www.pinecone.io/)
* Groq: [https://console.groq.com/](https://console.groq.com/)
* Cohere: [https://cohere.com/](https://cohere.com/)

---

### 📦 2. Preprocess Your Documents

* Split PDF files into chunks
* Generate embeddings using `all-MiniLM-L6-v2`
* Upload vectors to Pinecone

> ⚙️ Pinecone index configuration:
>
> * **Dimension**: 384
> * **Metric**: cosine

---

### 🚀 3. Deploy or Run Locally

#### ✅ Local Deployment

```bash
git clone https://github.com/rajput-musa/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file with your API keys:

```
PINECONE_API_KEY=your_key
PINECONE_ENV=your_env
PINECONE_INDEX=your_index

COHERE_API_KEY=your_key
GROQ_API_KEY=your_key
```

---

#### ☁️ Deploy on Hugging Face Spaces

1. Fork this repository or space
2. Add your API keys in **Settings → Secrets**
3. Update `app.py` with your Pinecone index and document references

