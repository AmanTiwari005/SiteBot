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
import time
from requests.exceptions import HTTPError
from datetime import datetime

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="SiteBot - Multi-Website Chat",
    page_icon="🤖",
    layout="wide"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Custom CSS for enhanced UI
def get_css(dark_mode: bool = False):
    base_css = """
        <style>
        /* General text and background */
        body {
            color: #333333;
            background-color: #ffffff;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #007bff;
            color: #ffffff;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Chat messages */
        .chat-message.user {
            padding: 12px;
            border-radius: 10px;
            margin: 8px 0;
            background-color: #007bff;
            color: #ffffff;
            font-size: 16px;
        }
        .chat-message.ai {
            padding: 12px;
            border-radius: 10px;
            margin: 8px 0;
            background-color: #28a745;
            color: #ffffff;
            font-size: 16px;
        }
        .timestamp {
            font-size: 12px;
            color: #cccccc;
            margin-top: 4px;
        }
        /* URL tags */
        .url-tag {
            display: inline-block;
            background-color: #28a745;
            color: #ffffff;
            padding: 6px 10px;
            border-radius: 12px;
            margin: 4px;
            font-size: 14px;
            font-weight: 500;
        }
        /* Headers and subheaders */
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        /* Input fields */
        .stTextInput input {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
    """
    dark_mode_css = """
        body {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #2c2c2c;
        }
        .chat-message.user {
            background-color: #1e90ff;
        }
        .chat-message.ai {
            background-color: #2ecc71;
        }
        .stTextInput input {
            background-color: #2c2c2c;
            color: #ffffff;
            border: 1px solid #555555;
        }
    """
    return base_css + (dark_mode_css if dark_mode else "") + "</style>"

class SiteBot:
    def __init__(self):
        """Initialize SiteBot with environment variables and model."""
        load_dotenv()
        self.llm_groq = self._init_groq_model()

    @st.cache_resource
    def _init_embeddings(_self):
        """Initialize and cache the embedding model."""
        logger.info("Initializing embedding model...")
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )

    @st.cache_resource
    def _init_groq_model(_self):
        """Initialize and cache the Groq model."""
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

    def _create_vectorstore(self, url: str) -> Chroma:
        """Create and return a vector store from a website URL with retry logic."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading content from {url}")
                loader = WebBaseLoader(url)
                documents = loader.load()

                if not documents:
                    raise ValueError(f"No content loaded from {url}")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                document_chunks = text_splitter.split_documents(documents)

                embeddings = self._init_embeddings()
                return Chroma.from_documents(document_chunks, embeddings)

            except HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit for {url}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries reached for {url}: {str(e)}")
                        raise Exception(f"Failed to process {url}: Too many requests to Hugging Face API.")
                else:
                    logger.error(f"HTTP error loading {url}: {str(e)}")
                    raise Exception(f"Failed to load {url}: {str(e)}")
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

    def generate_sample_queries(self, urls: List[str]) -> List[str]:
        """Generate sample queries based on loaded websites."""
        sample_queries = []
        for url in urls:
            sample_queries.append(f"What is the main topic of {url}?")
            sample_queries.append(f"Summarize the key points from {url}.")
        return sample_queries[:3]  # Limit to 3 for brevity

    def run(self) -> None:
        """Run the SiteBot application with an enhanced UI."""
        # Dark mode toggle
        dark_mode = st.checkbox("Dark Mode", value=False, key="dark_mode")
        st.markdown(get_css(dark_mode), unsafe_allow_html=True)

        # Collapsible sidebar
        show_sidebar = st.checkbox("Show Sidebar", value=True, key="show_sidebar")
        sidebar_container = st.sidebar if show_sidebar else st

        # Header
        st.header("🤖 SiteBot: Multi-Website Chat Assistant")
        st.markdown("Ask questions about multiple websites in one place!")

        # Sidebar content
        with sidebar_container:
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

        # Searchable chat history
        search_query = st.text_input("Search Chat History", placeholder="Type to filter messages...")
        filtered_history = [
            msg for msg in st.session_state.chat_history
            if not search_query.lower() or search_query.lower() in msg.content.lower()
        ]

        # Sample queries
        if st.session_state.get("website_urls"):
            with st.expander("Suggested Questions"):
                sample_queries = self.generate_sample_queries(st.session_state.website_urls)
                for query in sample_queries:
                    if st.button(query, key=f"sample_{query}"):
                        response = self.get_response(query, st.session_state.vector_stores, selected_urls)
                        st.session_state.chat_history.append(HumanMessage(content=query))
                        st.session_state.chat_history.append(AIMessage(content=response))
                        st.rerun()

        # Chat interface
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(filtered_history):
                with st.chat_message(
                    "assistant" if isinstance(message, AIMessage) else "user",
                    avatar="🤖" if isinstance(message, AIMessage) else "👤"
                ):
                    timestamp = datetime.now().strftime("%H:%M")
                    st.markdown(
                        f"<div class='chat-message {'ai' if isinstance(message, AIMessage) else 'user'}'>{message.content}</div>"
                        f"<div class='timestamp'>{timestamp}</div>",
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
                st.rerun()

        if user_query:
            response = self.get_response(user_query, st.session_state.vector_stores, selected_urls)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()

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
