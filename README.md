
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
    If you have Git installed:
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
    Create a `requirements.txt` file in your project root with the following content:
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
    Then, install the dependencies from your activated virtual environment:
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
```

**Step 2: Run the Chatbot Application**

Once the data ingestion is complete, you can start the chatbot interface.

Open your terminal in the project directory and run:
```bash
streamlit run app.py
```
This will typically open a new tab in your web browser, usually at http://localhost:8501. You can now interact with the chatbot by typing your questions into the input field.
