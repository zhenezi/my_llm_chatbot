# my_llm_chatbot
# Local Document Q&A Chatbot with Mistral & Langchain

This project implements a locally-run chatbot that allows you to ask questions about your Word (.docx) and PowerPoint (.pptx) documents. It uses a Retrieval Augmented Generation (RAG) approach, leveraging the Mistral Large Language Model (LLM) hosted locally via Ollama, Langchain for orchestration, ChromaDB for vector storage, and Streamlit for the user interface.

**Key Features:**

*   **Private & Offline:** All processing, including LLM inference, happens on your local machine. No data is sent to external services.
*   **Learns from Your Documents:** Ingests content from your specified Word and PowerPoint files.
*   **Natural Language Queries:** Ask questions in plain English and get answers based on the document content.
*   **Powered by Mistral:** Utilizes a powerful open-source LLM for response generation.
*   **Easy-to-Use Interface:** A simple web-based chat interface built with Streamlit.

## Project Structure
├── documents/ # <-- CREATE THIS: Place your .docx and .pptx files here

├── local_vector_db/ # Directory where the vector store will be created (auto-generated)

├── app.py # The Streamlit chatbot application

├── ingest_data.py # Script to process documents and create the vector store

└── README.md # This file

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python:** Version 3.8 or higher. ([Download Python](https://www.python.org/))
2.  **Ollama:** For running the Mistral LLM locally.
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   After installation, pull the Mistral model by running the following command in your terminal:
        ```bash
        ollama pull mistral
        ```
    *   Ensure the Ollama application/service is running in the background.

## Setup Instructions

1.  **Clone the Repository (or Download Files):**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
    Alternatively, download `app.py` and `ingest_data.py` into a new project directory.

2.  **Create and Activate a Virtual Environment (Recommended):**
    Open your terminal in the project directory:
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    *   On Windows (PowerShell/CMD):
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    langchain
    langchain_community
    ollama
    python-docx
    python-pptx
    unstructured
    sentence-transformers
    chromadb
    streamlit
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `unstructured` might require additional system dependencies like `libreoffice` or `pandoc` for certain file conversions, though it often works for basic .docx and .pptx without them. If you encounter issues with document loading, check the `unstructured` documentation.)*

4.  **Add Your Documents:**
    *   Create a folder named `documents` in the root of your project directory (if it doesn't exist).
    *   Place all your Word (`.docx`) and PowerPoint (`.pptx`) files that you want the chatbot to learn from into this `documents` folder.

## Running the Agent

There are two main steps to run the agent:

**Step 1: Ingest Your Documents (Data Processing)**

This step processes your documents, creates embeddings, and builds a local vector store. **You only need to run this once initially, and then again whenever you add, remove, or update files in the `documents` folder.**

Open your terminal (with the virtual environment activated) in the project directory and run:
```bash
python ingest_data.py
