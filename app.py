import streamlit as st
import os
import sys
import tempfile
import time # Optional: For simulating a loading time or delay

# --- LangChain and Google Imports ---
# NOTE: Removed getpass and userdata imports as they are replaced by st.secrets
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# Removed unused imports: ChatPromptTemplate, RunnablePassthrough, StrOutputParser

# ========================================================================
# 1. CONFIGURATION AND API KEY SETUP
# ========================================================================

# The PDF file name is now handled via Streamlit's file uploader (not fixed path)
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

st.set_page_config(page_title="Gemini Advanced RAG Chatbot", layout="wide")
st.title("🤖 Advanced Chat with Your PDF (Gemini PDR)")

# Streamlit requires the API key to be accessed via st.secrets during deployment.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("🚨 **Error:** GEMINI_API_KEY not found in Streamlit Secrets.")
        st.info("Please set this variable in your app's secrets.")
        st.stop()
except Exception:
    st.error("🚨 Configuration Error: Could not set Gemini API key.")
    st.stop()

# ========================================================================
# 2. RAG CORE LOGIC: PDR AND RETRIEVER (Cached Functions)
# ========================================================================

# Use st.cache_resource to ensure the expensive indexing step runs only once.
@st.cache_resource
def setup_retriever(file_path: str):
    # 1. Load Documents
    loader = PyPDFLoader(file_path)
    data = loader.load_and_split()

    # 2. Define Parent and Child Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    # 3. Setup Storage
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # We use a non-persistent vectorstore for the uploaded file in Streamlit
    vectorstore = Chroma.from_documents(data, embeddings)
    store = InMemoryStore()

    # 4. Create Parent Document Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    # This triggers the splitting and storage
    retriever.add_documents(data, ids=None)
    
    st.success(f"PDR indexing complete. {len(data)} pages processed.")
    return retriever

# ========================================================================
# 3. CONVERSATIONAL CHAIN WITH MEMORY (FIXED)
# ========================================================================

def setup_conversational_chain(retriever):
    # Define LLM and Memory
    # FIX: We now create the LLM object directly. 
    # The problematic line that referenced st.session_state is removed.
    llm_with_history = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2) 
    
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
    )
    # The function is much cleaner now!
    return qa_chain

# ========================================================================
# 3. STREAMLIT UI AND CHAT LOOP (MAIN EXECUTION)
# ========================================================================

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document to start chatting:", type="pdf")

if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF locally for the PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Setup the RAG components only when a file is uploaded
    with st.spinner("Indexing document with Parent Document Retriever..."):
        # Time the indexing for user feedback (optional)
        start_time = time.time()
        pdr_retriever = setup_retriever(tmp_file_path)
        qa_chain = setup_conversational_chain(pdr_retriever)
        end_time = time.time()
        st.success(f"Indexing complete in {end_time - start_time:.2f} seconds! Ask a multi-turn question below.")

    # Initialize chat history
    if "messages" not in st.session_state:
        # Initial greeting
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your uploaded PDF."}]
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking and retrieving context..."):
                # Invoke the Conversational Retrieval Chain
                # The chain automatically uses memory and PDR retriever
                result = qa_chain.invoke({"question": prompt})
                response = result['answer']
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
