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
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

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
            page_title="SiteBot - Website Chat Interface",
            page_icon="🤖",
            layout="wide"
        )
        st.title("SiteBot: Intelligent Website Chat")

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
            logger.error(f"Failed to create vector store: {str(e)}")
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

    def get_response(self, user_input: str, vector_store: Chroma) -> str:
        """Generate a response for the user input."""
        retriever_chain = self._get_context_retriever_chain(vector_store)
        rag_chain = self._get_conversational_rag_chain(retriever_chain)
        
        try:
            response = rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_input
            })
            return response['answer']
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Please try again."

    def run(self) -> None:
        """Run the SiteBot application."""
        with st.sidebar:
            st.header("Configuration")
            website_url = st.text_input("Enter Website URL", placeholder="https://example.com", key="website_url")
            # Store the previous URL in session state to detect changes
            if "prev_url" not in st.session_state:
                st.session_state.prev_url = None

        if not website_url:
            st.info("Please provide a valid website URL to begin.")
            return

        # Check if the URL has changed
        if st.session_state.prev_url != website_url:
            with st.spinner("Processing new website content..."):
                try:
                    # Update vector store and reset chat history for new URL
                    st.session_state.vector_store = self._create_vectorstore(website_url)
                    st.session_state.chat_history = [
                        AIMessage(content=f"Greetings! I've loaded {website_url}. How may I assist you today?")
                    ]
                    st.session_state.prev_url = website_url
                except Exception as e:
                    st.error(f"Failed to process the website: {str(e)}")
                    return

        # Initialize session state if not already done
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Greetings! I'm SiteBot. How may I assist you today?")
            ]
        if "vector_store" not in st.session_state:
            with st.spinner("Processing website content..."):
                st.session_state.vector_store = self._create_vectorstore(website_url)

        # Chat环球interface
        user_query = st.chat_input("Ask me anything about the website...")
        if user_query:
            response = self.get_response(user_query, st.session_state.vector_store)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Display conversation
        for message in st.session_state.chat_history:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.markdown(message.content)

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
