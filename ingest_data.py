import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "documents"
VECTOR_DB_DIR = "local_vector_db" # Directory to save the vector store

def load_documents():
    docs = []
    print(f"Loading documents from: {os.path.abspath(DOCS_DIR)}")
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".docx"):
            print(f"Loading .docx: {filename}")
            try:
                loader = UnstructuredWordDocumentLoader(filepath)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        elif filename.endswith(".pptx"):
            print(f"Loading .pptx: {filename}")
            try:
                loader = UnstructuredPowerPointLoader(filepath)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(docs)} document parts.")
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=200   # Overlap between chunks
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked_docs)} chunks.")
    return chunked_docs

def create_vector_store(chunked_documents):
    # Use a local embedding model
    # This will download the model on first run if not cached
    embedding_model_name = "all-MiniLM-L6-v2"
    print(f"Using embedding model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have a compatible GPU
    )

    # Create and persist the vector store
    print(f"Creating vector store in: {os.path.abspath(VECTOR_DB_DIR)}")
    vector_db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vector_db.persist() # Ensure data is saved to disk
    print("Vector store created and persisted.")
    return vector_db

if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"Error: The '{DOCS_DIR}' directory is empty or does not exist.")
        print("Please create it and add your .docx and .pptx files.")
    else:
        documents = load_documents()
        if documents:
            chunked_docs = split_documents(documents)
            create_vector_store(chunked_docs)
            print("Data ingestion complete. You can now run the chatbot app.")
        else:
            print("No documents were loaded. Please check your 'documents' folder and file types.")