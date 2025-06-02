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
