import streamlit as st
import os
import sys
import tempfile
import time

# --- LangChain and Google Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate # <-- Essential for custom prompt

# ========================================================================
# 1. CONFIGURATION AND API KEY SETUP
# ========================================================================

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

st.set_page_config(page_title="Gemini Advanced RAG Chatbot", layout="wide")
st.title("🤖 Advanced Chat with Your PDF (Gemini PDR)")

# API Key Setup via Streamlit Secrets
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("🚨 **Error:** GEMINI_API_KEY not found in Streamlit Secrets.")
        st.info("Please set this variable in your app's secrets.")
        st.stop()
except Exception:
    st.error("🚨 Configuration Error: Could not access Streamlit Secrets.")
    st.stop()


# ========================================================================
# 2. RAG CORE LOGIC: PDR AND RETRIEVER (Cached Function)
# ========================================================================

@st.cache_resource
def setup_retriever(file_path: str):
    # 1. Load Documents
    loader = PyPDFLoader(file_path)
    data = loader.load_and_split()

    # 2. Define Parent and Child Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    # 3. Setup Storage and Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(data, embeddings)
    store = InMemoryStore()

    # 4. Create Parent Document Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(data, ids=None)
    
    st.success(f"PDR indexing complete. {len(data)} pages processed.")
    return retriever

# ========================================================================
# 3. CONVERSATIONAL CHAIN WITH MEMORY AND CUSTOM PROMPT (QUALITY FIX)
# ========================================================================

# 1. Define the custom prompt template
CUSTOM_QA_TEMPLATE = """You are a highly analytical and concise research assistant. 
Your primary goal is to answer the user's question based ONLY on the provided context, 
which includes the conversation history and the retrieved document chunks.

Follow these rules STRICTLY:
1. **Conciseness:** Provide the answer directly and in a professional, factual tone. Avoid filler like "Based on the documents..."
2. **Refusal:** If the document does not contain the answer, state clearly: 'I could not find that specific detail in the document.'

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Concise Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_QA_TEMPLATE,
    input_variables=["context", "question", "chat_history"]
)


def setup_conversational_chain(retriever):
    # Define LLM and Memory
    llm_with_history = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.4)
    
    # Memory setup
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        input_key="question" 
    )

    # Create the final QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_with_history,
        retriever=retriever, 
        memory=memory,
        chain_type="stuff",
        # <-- CRITICAL FIX: Inject the custom prompt for quality control -->
        # combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT} 
    )
    st.info("Conversational Retrieval Chain setup complete with custom prompt.")
    return qa_chain

# ========================================================================
# 4. STREAMLIT UI AND CHAT LOOP (MAIN EXECUTION)
# ========================================================================

# Streamlit App Execution Flow

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document to start chatting:", type="pdf")

if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF locally for the PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Check if the chain is already initialized (to prevent re-indexing on every chat turn)
    if st.session_state.qa_chain is None:
        with st.spinner("Indexing document with Parent Document Retriever..."):
            start_time = time.time()
            pdr_retriever = setup_retriever(tmp_file_path)
            st.session_state.qa_chain = setup_conversational_chain(pdr_retriever)
            end_time = time.time()
            st.success(f"Indexing complete in {end_time - start_time:.2f} seconds! Ask a multi-turn question below.")

    # Initialize chat history for the session
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your uploaded PDF."}]
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking and retrieving context..."):
                # Invoke the Conversational Retrieval Chain
                # Use the chain from session state
                result = st.session_state.qa_chain.invoke({"question": prompt})
                response = result['answer']
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a PDF document to begin the RAG process and enable the chat.")
