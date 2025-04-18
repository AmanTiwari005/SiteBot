# SiteBot - Intelligent Web Assistant

SiteBot is a Streamlit-based web application that allows users to interact with website content intelligently. By loading website URLs, users can ask questions about the content, and SiteBot provides accurate, context-aware responses using Retrieval-Augmented Generation (RAG) powered by Groq and Chroma vector stores.

## Features
- **Website Content Analysis**: Load multiple website URLs and query their content.
- **Conversational Interface**: Engage in a chat-like interaction with the assistant, maintaining conversation history.
- **Suggested Queries**: Automatically generates relevant questions based on loaded website content.
- **Multi-Website Insights**: Combines information from multiple websites, highlighting agreements, differences, and unique perspectives.
- **Modern UI**: Built with Streamlit, Tailwind CSS, and custom styling for an enhanced user experience.
- **Robust Error Handling**: Gracefully handles network issues, invalid URLs, and other exceptions.

## Tech Stack
- **Python 3.8+**
- **Streamlit**: For the web interface.
- **LangChain**: For building the RAG pipeline.
- **Chroma**: For vector storage and retrieval.
- **Groq**: For language model inference.
- **HuggingFace Embeddings**: For text embeddings.
- **BeautifulSoup**: For web scraping.
- **Requests**: For HTTP requests.
- **ChromaDB**: For persistent vector storage.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sitebot.git
   cd sitebot
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the Application**:
   ```bash
   streamlit run app2.py
   ```

## Usage
1. **Add Websites**: In the sidebar, enter website URLs to load their content.
2. **Select Websites**: Choose which loaded websites to query.
3. **Ask Questions**: Use the chat input to ask questions about the selected websites.
4. **Explore Suggested Questions**: Try pre-generated questions for quick insights.
5. **Clear Chat**: Reset the conversation history using the "Clear Chat" button.

## Example
1. Add URLs like `https://example.com` and `https://anotherexample.com`.
2. Select both URLs from the multiselect dropdown.
3. Ask: "What are the main topics covered by these websites?"
4. SiteBot will provide a combined response with insights from both websites, including a comparative analysis if applicable.

## Project Structure
```
sitebot/
├── app2.py                # Main application script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not tracked)
├── chroma_db/             # Chroma vector store data
├── README.md              # Project documentation
```

## Dependencies
See `requirements.txt` for a complete list. Key dependencies include:
- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-groq`
- `chromadb`
- `requests`
- `beautifulsoup4`
- `python-dotenv`

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/) and [LangChain](https://langchain.com/).
- Powered by [Groq](https://groq.com/) for fast language model inference.
- Uses [Chroma](https://www.trychroma.com/) for efficient vector storage.