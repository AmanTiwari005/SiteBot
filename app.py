import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize GROQ chat model
def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    logger.info("GROQ_API_KEY loaded successfully.")
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.5
        )
    except Exception as e:
        logger.error(f"Failed to initialize Groq model: {str(e)}")
        raise

# Global model initialization with fallback
llm_groq = None
try:
    llm_groq = init_groq_model()
except Exception as e:
    st.error(f"Error initializing Groq model: {str(e)}. Please check your API key or contact Groq support.")
    logger.error(f"Model initialization failed: {str(e)}")

def get_vectorstore_from_url(url):
    try:
        # Load content from the URL
        loader = WebBaseLoader(url)
        document = loader.load()
        if not document:
            logger.warning(f"No content loaded from URL: {url}")
            return None
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        logger.info(f"Document split into {len(document_chunks)} chunks.")
        
        # Create a vector store
        embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(document_chunks, embeddings)
        logger.info(f"Vector store created with {vector_store._collection.count()} documents.")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store from {url}: {str(e)}")
        return None

def get_context_retriever_chain(vector_store):
    if not vector_store:
        logger.error("Vector store is None, cannot create retriever chain.")
        return None
    
    if llm_groq is None:
        logger.error("Language model (llm_groq) is not initialized.")
        return None
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up relevant information.")
    ])
    
    try:
        retriever_chain = create_history_aware_retriever(llm_groq, retriever, prompt)
        return retriever_chain
    except Exception as e:
        logger.error(f"Error creating retriever chain: {str(e)}")
        return None

def get_conversational_rag_chain(retriever_chain):
    if not retriever_chain:
        logger.error("Retriever chain is None, cannot create RAG chain.")
        return None
    
    if llm_groq is None:
        logger.error("Language model (llm_groq) is not initialized.")
        return None
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    try:
        stuff_documents_chain = create_stuff_documents_chain(llm_groq, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    except Exception as e:
        logger.error(f"Error creating conversational RAG chain: {str(e)}")
        return None

def get_response(user_input):
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        return "Error: Vector store not initialized. Please provide a valid website URL."
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    if not retriever_chain:
        return "Error: Failed to create retriever chain. Check logs for details."
    
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    if not conversation_rag_chain:
        return "Error: Failed to create conversation chain. Check logs for details."
    
    try:
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        logger.info(f"Response generated for input: {user_input}")
        return response['answer']
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {str(e)}")
        return f"Error: Unable to generate response. Details: {str(e)}"

# App config
st.set_page_config(page_title="SiteBot - Chat with Websites", page_icon="🤖")
st.title("SiteBot - Chat with Websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Main logic
if not website_url:
    st.info("Please enter a website URL")
else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        if not st.session_state.vector_store:
            st.error("Failed to load content from the provided URL. Please check the URL and try again.")

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
