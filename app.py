import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
from typing import Dict, List
import numpy as np
from asgiref.sync import sync_to_async
import django
from django.conf import settings
from django.db.models import Q
import asyncio

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project.settings")
django.setup()

from models import Collection, Document, Page, PageEmbedding  # Import Django models

# Set page config
st.set_page_config(
    page_title="DocBot - Document Chat",
    page_icon="📜",
    layout="wide"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
GROQ_MODEL = "llama-3.3-70b-versatile"

# Custom CSS
st.markdown("""
    <style>
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
    .chat-message {
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
        background-color: #f1f3f5;
        color: #333333;
        font-size: 16px;
    }
    .ai-message {
        background-color: #e9ecef;
    }
    .doc-tag {
        display: inline-block;
        background-color: #28a745;
        color: #ffffff;
        padding: 6px 10px;
        border-radius: 12px;
        margin: 4px;
        font-size: 14px;
        font-weight: 500;
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    .stTextInput input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

class DocBot:
    def __init__(self):
        """Initialize DocBot with Groq model."""
        load_dotenv()
        self.llm_groq = self._init_groq_model()

    def _init_groq_model(self) -> ChatGroq:
        """Initialize and return the Groq model."""
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found.")
            st.error("Please set your GROQ_API_KEY.")
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

    async def _get_similar_pages(self, query: str, document_ids: List[int]) -> List[dict]:
        """Perform similarity search using PageEmbedding."""
        # Simulate embedding the query (since models.py uses an external service)
        # For simplicity, assume query is converted to a 128-dim vector
        # In practice, you'd call the same embeddings service as in embed_document
        query_embedding = np.random.rand(128).tolist()  # Placeholder

        pages = await sync_to_async(PageEmbedding.objects.filter)(
            page__document_id__in=document_ids
        )
        pages = await sync_to_async(pages.annotate)(
            similarity=MaxSim('embedding', query_embedding)
        )
        pages = await sync_to_async(pages.order_by)('-similarity')[:3]
        
        results = []
        async for page_embedding in pages:
            page = await sync_to_async(getattr)(page_embedding, 'page')
            content = page.content or "Image-based page"
            results.append({
                "content": content,
                "document_name": page.document.name,
                "page_number": page.page_number,
                "similarity": page_embedding.similarity
            })
        return results

    def _get_context_retriever_chain(self):
        """Create a history-aware retriever chain."""
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Generate a search query based on the conversation.")
        ])
        
        # Custom retriever to use Django models
        class DjangoRetriever:
            def __init__(self, bot, document_ids):
                self.bot = bot
                self.document_ids = document_ids
            
            async def aget_relevant_documents(self, query):
                pages = await self.bot._get_similar_pages(query, self.document_ids)
                from langchain_core.documents import Document
                return [
                    Document(
                        page_content=page["content"],
                        metadata={
                            "document_name": page["document_name"],
                            "page_number": page["page_number"],
                            "similarity": page["similarity"]
                        }
                    )
                    for page in pages
                ]
        
        retriever = DjangoRetriever(self, st.session_state.get("selected_document_ids", []))
        return create_history_aware_retriever(self.llm_groq, retriever, prompt)

    def _get_conversational_rag_chain(self, retriever_chain):
        """Create a conversational RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on this context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        stuff_chain = create_stuff_documents_chain(self.llm_groq, prompt)
        return create_retrieval_chain(retriever_chain, stuff_chain)

    def get_response(self, user_input: str) -> str:
        """Generate a response based on selected documents."""
        if not st.session_state.get("selected_document_ids"):
            return "Please select at least one document to query."
        
        with st.spinner("Generating response..."):
            retriever_chain = self._get_context_retriever_chain()
            rag_chain = self._get_conversational_rag_chain(retriever_chain)
            try:
                response = rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_input
                })
                return response['answer']
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                return f"Sorry, I encountered an error: {str(e)}"

    async def run(self) -> None:
        """Run the DocBot application."""
        # Header
        st.header("📜 DocBot: Document Chat Assistant")
        st.markdown("Ask questions about your stored documents!")

        # Sidebar
        with st.sidebar:
            st.markdown("### Document Management")
            
            # Fetch collections
            collections = await sync_to_async(list)(
                Collection.objects.filter(owner__email=st.session_state.get("user_email", "default@domain.com"))
            )
            
            if not collections:
                st.info("No collections found. Please add documents via the Django backend.")
                return
            
            # Collection selection
            collection_names = [c.name for c in collections]
            selected_collection = st.selectbox(
                "Select Collection",
                options=collection_names,
                help="Choose a collection to view its documents."
            )
            
            if selected_collection:
                selected_collection_obj = next(c for c in collections if c.name == selected_collection)
                # Fetch documents
                documents = await sync_to_async(list)(
                    Document.objects.filter(collection=selected_collection_obj)
                )
                
                # Display documents
                st.markdown("#### Available Documents")
                for doc in documents:
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"<span class='doc-tag'>{doc.name}</span>", unsafe_allow_html=True)
                    if col2.button("✖", key=f"remove_{doc.id}", help=f"Remove {doc.name}"):
                        await sync_to_async(doc.delete)()
                        st.success(f"Removed {doc.name}")
                        st.rerun()
                
                # Document selection
                selected_documents = st.multiselect(
                    "Query These Documents",
                    options=[doc.name for doc in documents],
                    default=[],
                    help="Select one or more documents to include in your query."
                )
                
                # Update session state with selected document IDs
                st.session_state.selected_document_ids = [
                    doc.id for doc in documents if doc.name in selected_documents
                ]

        # Main content
        if not collections:
            st.info("Add documents via the Django backend to get started.")
            return

        # Chat history initialization
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm DocBot, your document assistant. Select documents in the sidebar and ask me anything.")
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
                    if isinstance(message, AIMessage) and i > 0:
                        if st.button("Copy", key=f"copy_{i}", help="Copy this response"):
                            st.write(f"Copied: {message.content}")

        # Chat input and clear button
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.chat_input("Ask about the selected documents...")
        with col2:
            if st.button("Clear Chat", help="Reset the conversation"):
                st.session_state.chat_history = [
                    AIMessage(content="Hello! I'm DocBot, your document assistant. Select documents in the sidebar and ask me anything.")
                ]
                st.rerun()

        if user_query:
            response = self.get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()

def main():
    """Main entry point."""
    try:
        bot = DocBot()
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.run())
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.critical(f"Application failed: {str(e)}")

if __name__ == "__main__":
    main()
