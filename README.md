# RAG AI Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Groq, designed to answer questions based on uploaded PDF documents. The chatbot uses a vector database for efficient document retrieval and the LLaMA3-8B model to generate contextually relevant responses. Users can upload multiple PDF files, manage them, and interact with the chatbot via a web-based interface.

## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features
- **PDF Document Processing**: Upload and process multiple PDF files to create a searchable knowledge base.
- **Document Retrieval**: Utilizes a FAISS vector database with HuggingFace embeddings for efficient retrieval of relevant document chunks.
- **Contextual Responses**: Generates accurate answers using the LLaMA3-8B model via Groq, based on retrieved document content.
- **File Management**: Allows users to upload and delete PDF files with a user-friendly interface.
- **Interactive Web Interface**: Built with Streamlit, featuring a chat interface and file management sidebar.
- **Persistent Chat History**: Maintains conversation history within a session using Streamlit's session state.
- **Error Handling**: Provides feedback for issues like missing files or vectorstore creation failures.

## Technologies
- **Python**: 3.8+
- **Streamlit**: For the web-based user interface.
- **LangChain**: For RAG pipeline, document loading, text splitting, and retrieval.
- **HuggingFace Embeddings**: `all-MiniLM-L12-v2` model for generating document embeddings.
- **Groq**: For accessing the LLaMA3-8B language model.
- **FAISS**: Vector database for efficient document retrieval.
- **PyPDFLoader**: For extracting text from PDF files.
- **python-dotenv**: For managing environment variables (e.g., Groq API key).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/rag-ai-chatbot.git
   cd rag-ai-chatbot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` file with the following:
   ```
   streamlit==1.39.0
   langchain==0.3.1
   langchain-groq==0.2.0
   langchain-community==0.3.1
   sentence-transformers==3.1.1
   faiss-cpu==1.8.0
   pypdf==5.0.1
   python-dotenv==1.0.1
   httpx==0.27.2
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Groq API key:
   ```bash
   GROQ_API_KEY=your_groq_api_key
   ```
   Obtain a Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys).

5. **Download HuggingFace Model**:
   The chatbot uses the `all-MiniLM-L12-v2` model for embeddings. The code assumes it’s stored at `C:/temp/Project Works/RAG PDF chatbot/all-MiniLM-L12-v2`. Update the path in the code or download the model:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')"
   ```
   Alternatively, modify the `model_name` path in `Ragbot.py` to a local or HuggingFace-hosted model.

## Usage

1. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run Ragbot.py
   ```

2. **Interact with the Chatbot**:
   - **Upload PDFs**: Use the left sidebar to upload one or more PDF files.
   - **Manage Files**: View uploaded files and delete them if needed.
   - **Ask Questions**: Enter queries in the chat input box. The chatbot retrieves relevant information from the uploaded PDFs and generates responses.
   - **View Chat History**: Previous messages are displayed in the chat interface.

3. **Example**:
   - Upload a PDF document (e.g., a research paper).
   - Ask: "What is the main topic of the document?"
   - The chatbot retrieves relevant sections and responds based on the content.

## Project Structure
```
rag-ai-chatbot/
│
├── Ragbot.py               # Main application script
├── .env                    # Environment variables (not tracked in git)
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── /temp/                  # Temporary directory for uploaded PDFs
```
