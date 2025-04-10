import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Custom CSS for professional styling
st.markdown("""
    <style>
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        background-color: #f1f3f5;
    }
    .ai-message {
        background-color: #e9ecef;
    }
    .url-tag {
        display: inline-block;
        background-color: #007bff;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
    }
    </style>
""", unsafe_allow_html=True)

class SiteBot:
    def __init__(self):
        """Initialize SiteBot with environment variables and model."""
        load_dotenv()
        self.llm_groq = self._init_groq_model()
        self._configure_streamlit()

    def _init_groq_model(self) -> ChatGroq:
        """Initialize and return the Groq model."""
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables.")
            st.error("Please set your GROQ_API_KEY in the environment.")
            raise ValueError("GROQ_API_KEY is required.")
        
        try:
            logger.info("Initializing Groq model...")
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=GROQ_MODEL,
                temperature=0.5
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq model: {str(e)}")
            raise

    def _configure_streamlit(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="SiteBot - Multi-Website Chat",
            page_icon="🤖",
            layout="wide"
        )

    def _create_vectorstore(self, url: str) -> Chroma:
        """Create and return a vector store from a website URL."""
        try:
            logger.info(f"Loading content from {url}")
            loader = WebBaseLoader(url)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            document_chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL)
            return Chroma.from_documents(document_chunks, embeddings)
        except Exception as e:
            logger.error(f"Failed to create vector store for {url}: {str(e)}")
            raise

    def _get_context_retriever_chain(self, vector_store: Chroma):
        """Create and return a context-aware retriever chain."""
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Generate a search query based on the conversation to retrieve relevant information.")
        ])
        
        return create_history_aware_retriever(self.llm_groq, retriever, prompt)

    def _get_conversational_rag_chain(self, retriever_chain):
        """Create and return a conversational RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide accurate answers based on this context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        stuff_chain = create_stuff_documents_chain(self.llm_groq, prompt)
        return create_retrieval_chain(retriever_chain, stuff_chain)

    def get_response(self, user_input: str, vector_stores: Dict[str, Chroma], selected_urls: List[str]) -> str:
        """Generate a response for the user input based on selected websites."""
        if not selected_urls:
            return "Please select at least one website to query."
        
        combined_context = ""
        with st.spinner("Generating response..."):
            for url in selected_urls:
                if url in vector_stores:
                    retriever_chain = self._get_context_retriever_chain(vector_stores[url])
                    rag_chain = self._get_conversational_rag_chain(retriever_chain)
                    try:
                        response = rag_chain.invoke({
                            "chat_history": st.session_state.chat_history,
                            "input": user_input
                        })
                        combined_context += f"\n\n**{url}**:\n{response['answer']}"
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
                        combined_context += f"\n\n**{url}**:\nSorry, I encountered an error."
                else:
                    combined_context += f"\n\n**{url}**:\nWebsite not loaded yet."
        
        return combined_context.strip()

    def run(self) -> None:
        """Run the SiteBot application with an enhanced UI."""
        # Header
        st.header("🤖 SiteBot: Multi-Website Chat Assistant")
        st.markdown("Ask questions about multiple websites in one place!")

        # Sidebar
        with st.sidebar:
            st.markdown("### Website Management")
            with st.form(key="url_form", clear_on_submit=True):
                new_url = st.text_input(
                    "Add a Website URL",
                    placeholder="https://example.com",
                    help="Enter a valid URL to load its content."
                )
                submit_button = st.form_submit_button(label="Add Website")

            if submit_button and new_url:
                if new_url not in st.session_state.get("website_urls", []):
                    with st.spinner(f"Loading {new_url}..."):
                        try:
                            st.session_state.setdefault("vector_stores", {})[new_url] = self._create_vectorstore(new_url)
                            st.session_state.setdefault("website_urls", []).append(new_url)
                            st.success(f"Successfully loaded {new_url}")
                        except Exception as e:
                            st.error(f"Failed to load {new_url}: {str(e)}")
                else:
                    st.warning(f"{new_url} is already loaded.")

            # Display loaded websites
            if st.session_state.get("website_urls"):
                st.markdown("#### Loaded Websites")
                for url in st.session_state.website_urls[:]:
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"<span class='url-tag'>{url}</span>", unsafe_allow_html=True)
                    if col2.button("✖", key=f"remove_{url}", help=f"Remove {url}"):
                        st.session_state.website_urls.remove(url)
                        st.session_state.vector_stores.pop(url, None)
                        st.success(f"Removed {url}")

            # Website selection
            selected_urls = st.multiselect(
                "Query These Websites",
                options=st.session_state.get("website_urls", []),
                default=st.session_state.get("website_urls", []),
                help="Select one or more websites to include in your query."
            )

        # Main content
        if not st.session_state.get("website_urls"):
            st.info("Add a website URL in the sidebar to get started.")
            return

        # Chat history initialization
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm SiteBot, your multi-website assistant. Add websites in the sidebar and ask me anything.")
            ]

        # Chat interface
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(
                    "assistant" if isinstance(message, AIMessage) else "user",
                    avatar="🤖" if isinstance(message, AIMessage) else "👤"
                ):
                    st.markdown(
                        f"<div class='chat-message {'ai-message' if isinstance(message, AIMessage) else ''}'>{message.content}</div>",
                        unsafe_allow_html=True
                    )
                    if isinstance(message, AIMessage) and i > 0:  # Skip welcome message
                        if st.button("Copy", key=f"copy_{i}", help="Copy this response"):
                            st.write(f"Copied: {message.content}")

        # Chat input and clear button
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.chat_input("Ask about the selected websites...")
        with col2:
            if st.button("Clear Chat", help="Reset the conversation"):
                st.session_state.chat_history = [
                    AIMessage(content="Hello! I'm SiteBot, your multi-website assistant. Add websites in the sidebar and ask me anything.")
                ]
                st.experimental_rerun()

        if user_query:
            response = self.get_response(user_query, st.session_state.vector_stores, selected_urls)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.experimental_rerun()

def main():
    """Main entry point for the application."""
    try:
        bot = SiteBot()
        bot.run()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.critical(f"Application failed: {str(e)}")

if __name__ == "__main__":
    main()
