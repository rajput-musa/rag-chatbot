%%writefile app.py

import streamlit as st
import os
from dotenv import load_dotenv

# --- Core LangChain and Service Imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- Advanced Re-Ranking Imports ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# --- Load Environment Variables from .env file ---
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY # Set for Pinecone client
INDEX_NAME = "rag-chatbot" # Make sure this matches your Pinecone index

# --- Page Configuration ---
st.set_page_config(page_title="Production RAG System", page_icon="🚀", layout="wide")
st.title("🚀 Production-Grade RAG System")
st.write("This app uses a state-of-the-art RAG pipeline with Pinecone, Groq, and Cohere Re-ranking for fast, accurate, and structured answers.")

# --- Pydantic Model for Structured Output ---
class StructuredAnswer(BaseModel):
    summary: str = Field(description="A concise, synthesized summary of the answer based on the context.")
    key_points: list[str] = Field(description="A list of 2-4 key bullet points that support the summary.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating the model's confidence that the answer is fully supported by the provided context.")

# --- Caching and Model/Service Initialization ---
@st.cache_resource
def initialize_services():
    """Initializes and caches all the necessary services and models."""
    # Check for API keys
    if not all([PINECONE_API_KEY, GROQ_API_KEY, COHERE_API_KEY]):
        raise ValueError("API keys for Pinecone, Groq, or Cohere are missing.")

    with st.spinner("Initializing services... (this happens once)"):
        # Embedding model (still runs locally but is very fast)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        
        # Pinecone Vector Store
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        
        # Base Retriever from Pinecone
        base_retriever = vectorstore.as_retriever(search_kwargs={'k': 20}) # Retrieve a large pool of docs
        
        # Cohere Re-ranker
        compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=5) # Intelligently pick the top 5
        
        # Advanced Re-ranking Retriever
        reranking_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        
        # Groq LLM
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY) # Use the more powerful 70B model
        
        return reranking_retriever, llm

# --- Main Application Logic ---
try:
    retriever, llm = initialize_services()
except ValueError as e:
    st.error(str(e))
    st.stop()

# --- RAG Chain Definition ---
pydantic_parser = PydanticOutputParser(pydantic_object=StructuredAnswer)
format_instructions = pydantic_parser.get_format_instructions()

template = """
You are a world-class analysis engine. Your task is to provide a structured, factual answer based *only* on the following context.
Synthesize the information from all context snippets. Do not use any outside knowledge.

Context:
{context}

Question:
{question}

Follow these formatting instructions precisely:
{format_instructions}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions}
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | pydantic_parser
)

st.success("System is ready. Ask your question below.")
query = st.text_input("Enter your question:", key="query_input")

if query:
    with st.spinner("Searching, re-ranking, and generating a high-quality answer..."):
        try:
            structured_answer = rag_chain.invoke(query)
            
            st.write("### Answer")
            st.markdown(f"**Summary:** {structured_answer.summary}")
            st.markdown(f"**Confidence:**")
            st.progress(structured_answer.confidence_score)
            
            st.markdown("**Key Points:**")
            for point in structured_answer.key_points:
                st.markdown(f"- {point}")
            
            with st.expander("Show Re-ranked Sources"):
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    source_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page_number = doc.metadata.get('page', 'N/A')
                    st.markdown(f"**Source:** `{source_filename}` (Page: {page_number})")
                    st.markdown(f"> {doc.page_content}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")
