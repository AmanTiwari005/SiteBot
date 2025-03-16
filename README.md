# 🤖 SiteBot: Chat with Websites

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMa--3.3--70B-green)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

SiteBot is an AI-powered chatbot that allows you to have natural conversations with any website. Simply enter a URL, and SiteBot will scrape the content, process it, and enable you to ask questions about the website's information. Built with Streamlit, LangChain, and Groq's LLaMa 3.3 model.

![SiteBot Demo](https://raw.githubusercontent.com/yourusername/sitebot/main/demo.png)

## ✨ Features

- 🌐 **Website Scraping**: Extract content from any website URL
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 💬 **Conversational AI**: Chat naturally with the website content
- 🧠 **Context-Aware**: Maintains conversation history for coherent responses
- ⚡ **Fast Processing**: Powered by Groq's LLaMa 3.3 70B model

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sitebot.git
cd sitebot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 Environment Setup

1. Create a `.env` file in the project root
2. Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### How to use:
1. Enter a website URL in the sidebar
2. Wait for the content to be processed
3. Start chatting with the website!

## 📋 Requirements

- Python 3.8+
- Streamlit
- LangChain
- Groq API access
- HuggingFace Transformers
- Chroma DB
- dotenv

## 🔄 How It Works

1. **Content Extraction**: Scrapes website content using WebBaseLoader
2. **Text Processing**: Splits content into manageable chunks
3. **Embedding Generation**: Creates vector embeddings using HuggingFace's Instruct Embeddings
4. **Vector Storage**: Stores embeddings in Chroma DB for efficient retrieval
5. **Conversational Chain**: Uses LangChain to create a conversational RAG (Retrieval Augmented Generation) system
6. **Response Generation**: Generates contextually relevant responses using Groq's LLaMa 3.3 model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [LangChain](https://www.langchain.com/) for the RAG implementation
- [Groq](https://groq.com/) for the LLM API
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Chroma DB](https://www.trychroma.com/) for vector storage

---

Made with ❤️ by Aman Tiwari
