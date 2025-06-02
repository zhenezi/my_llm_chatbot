import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

VECTOR_DB_DIR = "local_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral" # Ensure this model is pulled in Ollama

# --- Load resources (cached for performance) ---
@st.cache_resource # Use cache_resource for non-data objects
def load_llm():
    print(f"Loading LLM: {OLLAMA_MODEL_NAME}")
    try:
        llm = Ollama(model=OLLAMA_MODEL_NAME)
        # Test invocation
        llm.invoke("Test")
        print("LLM loaded successfully.")
        return llm
    except Exception as e:
        st.error(f"Failed to load Ollama LLM ({OLLAMA_MODEL_NAME}). Is Ollama running and the model pulled? Error: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Change to 'cuda' for GPU
        )
        print("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model. Error: {e}")
        return None

@st.cache_resource
def load_vector_store(_embeddings): # Pass embeddings to ensure it's loaded first
    if not os.path.exists(VECTOR_DB_DIR):
        st.error(f"Vector store not found at {VECTOR_DB_DIR}. Please run ingest_data.py first.")
        return None
    print(f"Loading vector store from: {VECTOR_DB_DIR}")
    try:
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=_embeddings
        )
        print("Vector store loaded successfully.")
        return vector_db
    except Exception as e:
        st.error(f"Failed to load vector store. Error: {e}")
        return None

# --- RAG Chain Setup ---
def get_qa_chain(llm, vector_store):
    if llm is None or vector_store is None:
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    prompt_template = """You are an AI assistant helping answer questions based on the provided context.
    Use only the information from the context to answer the question.
    If the information is not in the context, say "I don't have information on that in the provided documents."
    Be concise and helpful.

    Context:
    {context}

    Question: {question}

    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all context in one prompt
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True # Optional: to see which chunks were used
    )
    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Local Document Chatbot", layout="wide")
st.title("📝 Local Document Chatbot (Mistral + Your Docs)")
st.caption("Ask questions about the documents you provided during ingestion.")

# Load resources
llm = load_llm()
embeddings = load_embedding_model()
vector_store = None
if embeddings: # Only load vector store if embeddings loaded
    vector_store = load_vector_store(embeddings)

qa_chain = None
if llm and vector_store:
    qa_chain = get_qa_chain(llm, vector_store)

if not qa_chain:
    st.warning("Chatbot is not ready. Please check error messages above and ensure Ollama is running, models are pulled, and data ingestion was successful.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you based on your documents?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask your question...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke({"query": user_query})
                    answer = response.get("result", "Sorry, I couldn't find an answer.")
                    message_placeholder.markdown(answer)

                    # Optional: Display source documents
                    with st.expander("Show Sources"):
                        if "source_documents" in response and response["source_documents"]:
                            for i, doc in enumerate(response["source_documents"]):
                                st.write(f"**Source {i+1}:**")
                                st.caption(f"File: {doc.metadata.get('source', 'N/A')}") # Assuming 'source' is in metadata
                                st.text_area(f"Content Snippet {i+1}", doc.page_content[:500] + "...", height=100, key=f"source_{i}")
                        else:
                            st.write("No specific source documents were retrieved for this answer.")

                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    answer = "Sorry, an error occurred while processing your request."
                    message_placeholder.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})