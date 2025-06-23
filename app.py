import streamlit as st
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/app/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/huggingface_cache/transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/huggingface_cache/sentence_transformers'
if not os.path.exists('/app/huggingface_cache'):
    os.makedirs('/app/huggingface_cache', exist_ok=True)

import langchain

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank


try:
    print("Step 1: Loading environment variables...")
    load_dotenv()
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    INDEX_NAME = "rag-chatbot"
    print("Step 1: SUCCESS")

    st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🚀", layout="wide")
    

    st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                width: 100%;
                margin: auto; /* Can adjust max-width if needed */
            }
            .chat-row {
                display: flex;
                align-items: flex-start;
                margin-bottom: 1rem;
                width: 100%;
            }
            .chat-bubble {
                padding: 0.9rem;
                border-radius: 1rem;
                max-width: 70%;
                word-wrap: break-word;
                font-size: 1rem;
                line-height: 1.4;
            }
            .user-row {
                justify-content: flex-end;
            }
            .bot-row {
                justify-content: flex-start;
            }
            .user-bubble {
                background-color: #0078D4; /* Streamlit blue */
                color: white;
                border-bottom-right-radius: 0.2rem;
            }
            .bot-bubble {
                background-color: #262730; /* Streamlit dark theme component background */
                color: white;
                border-bottom-left-radius: 0.2rem;
                border: 1px solid #3c3d49; /* Slight border for bot bubble */
            }
            .avatar {
                font-size: 1.5rem;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                background-color: #4A4A4A; /* Neutral avatar background */
            }
            .user-avatar { margin-left: 0.5rem; }
            .bot-avatar { margin-right: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def initialize_services():
        
        print("Step 2: Entering initialize_services function...")
        if not all([PINECONE_API_KEY, GROQ_API_KEY, COHERE_API_KEY]):
            raise ValueError("An API key is missing!")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        host = "https://rag-chatbot-sg8t88c.svc.aped-4627-b74a.pinecone.io" 
        index = pinecone.Index(host=host)
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        base_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
        reranking_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
        print("Step 2: All services initialized successfully.")
        return reranking_retriever, llm

    print("Step 3: Calling initialize_services...")
    retriever, llm = initialize_services()
    print("Step 3: SUCCESS, services are loaded.")

    # --- RAG CHAIN 
    print("Step 4: Defining RAG chain...")
    system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context.
    Your answer should be concise and directly address the question.
    After your answer, list the numbers of the sources you used, like this: [1][2].
    Do not make up information. If the answer is not in the context, say "I cannot answer this based on the provided documents."
    Context (Sources are numbered starting from 1):
    {context}
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs_with_numbers(docs):
        numbered_docs = []
        MAX_DOC_LENGTH = 1200 # Adjusted max length
        for i, doc in enumerate(docs):
            content = doc.page_content
            if len(content) > MAX_DOC_LENGTH:
                content = content[:MAX_DOC_LENGTH] + "..."
            numbered_docs.append(f"Source [{i+1}]:\n{content}")
        return "\n\n".join(numbered_docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_numbers), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    print("Step 4: SUCCESS")

    # --- Streamlit Chat UI
    st.title("💬 Document Chatbot Interface") 

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to answer questions about your documents.", "sources": []}]

   
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-row user-row"><div class="chat-bubble user-bubble">{message["content"]}</div><div class="avatar user-avatar">🤔</div></div>', unsafe_allow_html=True)
        else: # Assistant
            st.markdown(f'<div class="chat-row bot-row"><div class="avatar bot-avatar">🤖</div><div class="chat-bubble bot-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
            if message.get("sources"): # Check if sources exist for this message
                with st.expander("Sources Referenced in this Answer"):
                    for i, doc_info in enumerate(message["sources"]):
                        st.markdown(f"**[{i+1}] Source:** `{doc_info['filename']}` (Page: {doc_info['page']})")
                        st.markdown(f"> {doc_info['content_snippet'][:300]}...") # Show snippet
                        st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)


    if user_query := st.chat_input("Ask a question about your documents"):
        # Add user message to history and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.markdown(f'<div class="chat-row user-row"><div class="chat-bubble user-bubble">{user_query}</div><div class="avatar user-avatar">🤔</div></div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="🤖"): # Use Streamlit's chat_message context for the spinner and bot response area
            message_placeholder = st.empty() # Create a placeholder for the streaming response
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    print(f"--- UI DEBUG: Invoking RAG chain with query: {user_query} ---")
                    
                   
                    assistant_response_content = ""
                    for chunk in rag_chain.stream(user_query):
                        assistant_response_content += chunk
                        message_placeholder.markdown(f'<div class="chat-bubble bot-bubble">{assistant_response_content}▌</div>', unsafe_allow_html=True) # Typing effect
                    
                    message_placeholder.markdown(f'<div class="chat-bubble bot-bubble">{assistant_response_content}</div>', unsafe_allow_html=True) # Final response
                    print(f"--- UI DEBUG: Full LLM Answer: {assistant_response_content} ---")

                    
                    retrieved_docs_for_display = retriever.invoke(user_query)
                    sources_info_for_display = []
                    if retrieved_docs_for_display:
                        for doc in retrieved_docs_for_display:
                            sources_info_for_display.append({
                                "filename": os.path.basename(doc.metadata.get('source', 'Unknown')),
                                "page": doc.metadata.get('page', 'N/A'),
                                "content_snippet": doc.page_content
                            })
                    
                   
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response_content,
                        "sources": sources_info_for_display 
                    })

                    
                    if sources_info_for_display:
                        with st.expander("Sources for the latest answer"):
                            for i, doc_info in enumerate(sources_info_for_display):
                                st.markdown(f"**[{i+1}] Source:** `{doc_info['filename']}` (Page: {doc_info['page']})")
                                st.markdown(f"> {doc_info['content_snippet'][:300]}...")
                                st.markdown("---")
                    
                    

                except Exception as e_invoke:
                    error_message = f"Error processing your query: {e_invoke}"
                    print(f"!!!!!!!!!! ERROR DURING RAG CHAIN INVOCATION (UI Level) !!!!!!!!!!")
                    import traceback
                    print(traceback.format_exc())
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error_message}", "sources": []})
    
    print("--- app.py script finished a run ---")
    
except Exception as e:
    print(f"!!!!!!!!!! A FATAL ERROR OCCURRED DURING STARTUP !!!!!!!!!!")
    import traceback
    print(traceback.format_exc())
    st.error(f"A fatal error occurred during startup. Please check the container logs. Error: {e}")
