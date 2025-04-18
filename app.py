from typing import List, Dict, Tuple, Optional
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
import time
from groq import Groq
from requests.exceptions import HTTPError, RequestException
from datetime import datetime
import chromadb
from bs4 import BeautifulSoup
import hashlib
import re
from urllib.parse import urlparse
from collections import Counter
from typing import List, Dict, Tuple
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container

load_dotenv()
# Set page config with better aesthetics
st.set_page_config(
    page_title="SiteBot - Intelligent Web Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-70b-8192"  # Updated to current Groq model name

class SiteBot:
    def __init__(self):
        """Initialize SiteBot with environment variables and model."""
        load_dotenv()
        self.llm_groq = self._init_groq_model()

    @st.cache_resource
    def _init_embeddings(_self):
        """Initialize and cache the embedding model."""
        logger.info("Initializing embedding model...")
        return HuggingFaceEmbeddings(
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

    def _load_documents(self, url: str) -> List[Document]:
        """Load documents using WebBaseLoader with improved settings."""
        try:
            logger.info(f"Loading content from {url}...")
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
                "verify": True,
                "allow_redirects": True,
                "timeout": 60
            }
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content loaded from {url}")
                return [Document(
                    page_content="Unable to load content. The page may require authentication or be restricted.",
                    metadata={"source": url, "section": "fallback"}
                )]
            
            return documents
        except Exception as e:
            logger.error(f"Failed to load {url}: {str(e)}")
            return [Document(
                page_content=f"Failed to load content: {str(e)}",
                metadata={"source": url, "section": "fallback"}
            )]

    def _process_large_document(self, document: Document) -> List[Document]:
        """Process large documents by splitting them into manageable chunks."""
        content = document.page_content
        metadata = document.metadata
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents([document])

    def _create_vectorstore(self, url: str) -> Chroma:
        """Create vector store from website content."""
        max_retries = 3
        retry_delay = 5
        collection_name = f"sitebot_{hashlib.md5(url.encode()).hexdigest()}"
        
        for attempt in range(max_retries):
            try:
                documents = self._load_documents(url)
                processed_docs = []
                for doc in documents:
                    processed_docs.extend(self._process_large_document(doc))
                
                if not any(doc.page_content.strip() for doc in processed_docs):
                    raise ValueError(f"Loaded documents from {url} contain no usable content.")

                embeddings = self._init_embeddings()
                
                chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                try:
                    chroma_client.delete_collection(collection_name)
                except Exception:
                    pass
                
                return Chroma.from_documents(
                    processed_docs,
                    embeddings,
                    client=chroma_client,
                    collection_name=collection_name,
                    collection_metadata={"url": url, "timestamp": str(datetime.now())}
                )

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    def _get_context_retriever_chain(self, vector_store: Chroma):
        """Create retriever chain that's aware of conversation history."""
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given this conversation, generate a search query to find relevant information")
        ])
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return create_history_aware_retriever(self.llm_groq, retriever, prompt)

    def _get_conversational_rag_chain(self, retriever_chain):
        """Create conversational RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer questions based on the provided website content. 
             If you don't know, say so. Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        stuff_chain = create_stuff_documents_chain(self.llm_groq, prompt)
        return create_retrieval_chain(retriever_chain, stuff_chain)

    def get_response(self, user_input: str, vector_stores: Dict[str, Chroma], selected_urls: List[str]) -> str:
        """Generate unified response combining relevant information from all selected websites."""
        if not selected_urls:
            return "Please select at least one website to query."
        
        responses = []
        relevant_sources = set()
        
        # First pass: Identify relevant websites for the question
        for url in selected_urls:
            if url in vector_stores:
                try:
                    retriever = vector_stores[url].as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(user_input)
                    if relevant_docs:
                        relevant_sources.add(url)
                except Exception as e:
                    logger.error(f"Error checking relevance for {url}: {str(e)}")
        
        # If no obviously relevant sites, use all selected sites
        if not relevant_sources:
            relevant_sources = set(selected_urls)
        
        # Second pass: Get detailed answers from relevant sites
        for url in relevant_sources:
            try:
                retriever_chain = self._get_context_retriever_chain(vector_stores[url])
                rag_chain = self._get_conversational_rag_chain(retriever_chain)
                
                response = rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_input
                })
                
                if response['answer'].strip():
                    domain = urlparse(url).netloc.replace("www.", "").split('.')[0].title()
                    responses.append({
                        "source": domain,
                        "url": url,
                        "content": response['answer']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                responses.append({
                    "source": "Error",
                    "url": url,
                    "content": f"⚠️ Could not retrieve information: {str(e)}"
                })
        
        # Generate unified response
        if not responses:
            return "No relevant information found across the selected websites."
        
        # For single-source responses
        if len(responses) == 1:
            return f"**From {responses[0]['source']}:**\n{responses[0]['content']}"
        
        # For multi-source responses
        answer_parts = []
        comparison_notes = []
        all_contents = [r['content'] for r in responses]
        
        # Look for agreements/disagreements between sources
        if len(responses) > 1:
            comparison_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze these different responses about the same topic:
                {contents}
                
                Identify:
                1. Key points of agreement
                2. Important differences
                3. Unique perspectives from each source"""),
            ])
            
            comparison_chain = comparison_prompt | self.llm_groq
            comparison_result = comparison_chain.invoke({"contents": "\n\n".join(all_contents)})
            comparison_notes.append(str(comparison_result.content))
        
        # Build final response
        answer_parts.append("### Combined Insights from Multiple Websites")
        
        for response in responses:
            answer_parts.append(f"""
    **From {response['source']}** ([source]({response['url']})):
    {response['content']}
    """)
        
        if comparison_notes:
            answer_parts.append("""
    ### Comparative Analysis
    """ + "\n".join(comparison_notes))
        
        return "\n".join(answer_parts)

    def _extract_key_terms(self, documents: List[Document]) -> List[str]:
        """Extract key terms from documents."""
        text = " ".join(doc.page_content for doc in documents)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'and', 'the', 'of', 'in', 'to', 'a', 'is', 'for', 'on', 'with'}
        keywords = [word for word in words if word not in stop_words]
        return [word for word, _ in Counter(keywords).most_common(5)]

    def generate_sample_queries(self, urls: List[str], vector_stores: Dict[str, Chroma]) -> List[Tuple[str, str]]:
        """Generate balanced sample queries across all loaded websites."""
        sample_queries = []
        max_queries_per_site = 2  # Maximum queries to generate per website
        
        for url in urls:
            if url not in vector_stores:
                continue
                
            try:
                # Extract domain and potential topic keywords
                domain = urlparse(url).netloc.replace("www.", "").split('.')[0].title()
                documents = vector_stores[url].get()['documents']
                docs = [Document(page_content=doc) for doc in documents]
                terms = self._extract_key_terms(docs)
                
                # Generate queries for this URL
                url_queries = []
                
                if terms:
                    url_queries.extend([
                        (f"What does this page say about {terms[0]}?", url),
                        (f"Explain the concept of {terms[0]}", url),
                        (f"How does {domain} approach {terms[0]}?", url),
                        (f"Summarize the key points about {terms[0]}", url),
                    ])
                else:
                    url_queries.extend([
                        (f"What is the main purpose of {domain}?", url),
                        ("Summarize the key points from this content", url),
                    ])
                
                # Add domain-specific queries
                url_queries.extend([
                    (f"What can I learn from {domain}'s content?", url),
                    (f"What are the main topics covered by {domain}?", url),
                ])
                
                # Add a balanced selection from this URL's queries
                sample_queries.extend(url_queries[:max_queries_per_site])
                
            except Exception as e:
                logger.warning(f"Couldn't generate sample queries for {url}: {str(e)}")
                sample_queries.extend([
                    ("What is this page about?", url),
                    ("Summarize this content", url),
                ][:max_queries_per_site])
        
        # Shuffle to mix queries from different sites
        import random
        random.shuffle(sample_queries)
        
        return sample_queries[:6]  # Return top 6 mixed queries
            
        return unique_queries[:6]  # Return top 6 unique queries
    def run(self):
        """Run the enhanced SiteBot application."""
        # Inject Tailwind CSS and custom styles
        st.markdown("""
                <style>
    /* Custom scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
      /* Fix button focus states */
    button[data-baseweb="button"]:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.5);
    }
    
    /* Better card hover effects */
    div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"] {
        transition: all 0.2s ease;
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"]:hover {
        border-color: #6366f1 !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Darker version for sidebar */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
    }
  
    button:active {
        transform: scale(0.98);
    }

    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    </style>
""", unsafe_allow_html=True)

        # Initialize session state variables if they don't exist
        if "website_urls" not in st.session_state:
            st.session_state["website_urls"] = []
        if "vector_stores" not in st.session_state:
            st.session_state["vector_stores"] = {}
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                AIMessage(content="Hello! I'm SiteBot, your web assistant. Add websites in the sidebar and ask me anything about their content.")
            ]

        # Hero Section
        with st.container():
            colored_header(
                label="🌐 SiteBot - Your Intelligent Web Assistant",
                description="Chat with any website's content",
                color_name="violet-70"
            )
            st.markdown("""
            <div style="text-align: center; margin-bottom: 32px;">
                <p style="font-size: 18px; color: #6b7280;">
                    Add website URLs and get instant answers from their content
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Sidebar with improved layout
        with st.sidebar:
            st.markdown("""
            <div style="margin-bottom: 32px;">
                <h2 style="color: white; font-weight: 700;">Website Management</h2>
                <p style="color: #e5e7eb;">Add and manage websites to query</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get existing website URLs before modifying
            website_urls = list(st.session_state.get("website_urls", []))
            
            with st.form(key="url_form", clear_on_submit=True):
                new_url = st.text_input(
                    "Add Website URL",
                    placeholder="https://example.com",
                    help="Enter a valid URL to analyze"
                )
                submit_button = st.form_submit_button(
                    "Add Website", 
                    type="primary",
                    use_container_width=True
                )

            if submit_button and new_url:
                if new_url not in website_urls:
                    with st.spinner(f"🌐 Loading {new_url}..."):
                        try:
                            vector_stores = dict(st.session_state.get("vector_stores", {}))
                            vector_stores[new_url] = self._create_vectorstore(new_url)
                            st.session_state["vector_stores"] = vector_stores
                            
                            # Update website URLs separately to prevent UI issues
                            website_urls.append(new_url)
                            st.session_state["website_urls"] = website_urls
                            
                            st.success(f"✅ Successfully loaded {new_url}")
                        except Exception as e:
                            st.error(f"❌ Failed to load {new_url}: {str(e)}")
                else:
                    st.warning(f"⚠️ {new_url} is already loaded")

            # Display loaded websites
            website_urls = list(st.session_state.get("website_urls", []))
            if website_urls:
                st.markdown("""
                <div style="margin-top: 32px;">
                    <h3 style="color: white; font-weight: 600;">Loaded Websites</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a copy of website_urls to safely iterate
                urls_to_display = website_urls.copy()
                urls_to_remove = []
                
                for url in urls_to_display:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        col1.markdown(f'<span class="url-tag">{url}</span>', unsafe_allow_html=True)
                        if col2.button("✖", key=f"remove_{url}"):
                            urls_to_remove.append(url)
                
                # Remove URLs outside the loop to prevent UI issues
                if urls_to_remove:
                    for url in urls_to_remove:
                        if url in website_urls:
                            website_urls.remove(url)
                            st.session_state.vector_stores.pop(url, None)
                    
                    st.session_state["website_urls"] = website_urls
                    st.rerun()

            # Updated website list after any removals
            website_urls = list(st.session_state.get("website_urls", []))
            selected_urls = st.multiselect(
                "Websites to Query",
                options=website_urls,
                default=website_urls,
                help="Select websites to include in your query"
            )

        # Main content area
        if not st.session_state.get("website_urls"):
            st.markdown("""
            <div style="text-align: center; padding: 64px 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <h3 style="color: #6b7280; margin-top: 16px;">Add website URLs to get started</h3>
                <p style="color: #9ca3af;">Use the sidebar to add websites you want to query</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Suggested questions section
        if st.session_state.get("website_urls"):
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <h3 style="color: #1f2937; font-weight: 600;">Suggested Questions</h3>
                <p style="color: #6b7280;">Try these questions about your loaded websites</p>
            </div>
            """, unsafe_allow_html=True)
            
            website_urls = list(st.session_state.get("website_urls", []))
            vector_stores = dict(st.session_state.get("vector_stores", {}))
            
            try:
                sample_queries = self.generate_sample_queries(website_urls, vector_stores)
                
                # Create a row for each pair of suggestions (2 per row)
                for i in range(0, len(sample_queries), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(sample_queries):
                            query, url = sample_queries[i + j]
                            with cols[j]:
                                with stylable_container(
                                    key=f"container_{i}_{j}",
                                    css_styles="""
                                    {
                                        border: 1px solid rgba(49, 51, 63, 0.2);
                                        border-radius: 0.5rem;
                                        padding: calc(1em - 1px);
                                        margin-bottom: 1rem;
                                    }
                                    """
                                ):
                                    st.markdown(f"""
                                    <div style="font-size: 14px; margin-bottom: 8px;">
                                        {query}
                                    </div>
                                    <div style="font-size: 12px; color: #9ca3af;">
                                        From: {urlparse(url).netloc}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if st.button(
                                        "Ask", 
                                        key=f"sample_btn_{hashlib.sha256(f'{query}_{url}'.encode()).hexdigest()}",
                                        use_container_width=True
                                    ):
                                        # Process the query
                                        response = self.get_response(
                                            user_input=query,
                                            vector_stores=vector_stores,
                                            selected_urls=website_urls
                                        )
                                        
                                        # Update chat history
                                        st.session_state.chat_history.extend([
                                            HumanMessage(content=query),
                                            AIMessage(content=response)
                                        ])
                                        
                                        # Force rerun
                                        st.rerun()

            except Exception as e:
                st.error(f"Error generating suggested questions: {str(e)}")

        # Chat history display
        st.markdown("""
        <div style="margin: 32px 0 16px 0;">
            <h3 style="color: #1f2937; font-weight: 600;">Conversation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get a copy of chat history to prevent modification during display
        chat_history = list(st.session_state.chat_history)
        
        for message in chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🌐"):
                    st.markdown(f'<div class="bot-message">{message.content}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f'<div class="user-message">{message.content}</div>', unsafe_allow_html=True)

        # Input area at bottom
        with st.container():
            input_col, button_col = st.columns([5, 1])
            with input_col:
                user_query = st.chat_input("Ask about the selected websites...", key="user_query")
            with button_col:
                if st.button("Clear Chat", use_container_width=True, type="secondary"):
                    st.session_state.chat_history = [
                        AIMessage(content="Hello! I'm SiteBot, your web assistant. What would you like to know about your loaded websites?")
                    ]
                    st.rerun()

        if user_query:
            # Store current vector stores and selected URLs to prevent modification during processing
            vector_stores = dict(st.session_state.get("vector_stores", {}))
            selected_urls_current = list(selected_urls)
            
            # Generate response
            response = self.get_response(user_query, vector_stores, selected_urls_current)
            
            # Update chat history
            chat_history = list(st.session_state.chat_history)
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=response))
            st.session_state.chat_history = chat_history
            
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
