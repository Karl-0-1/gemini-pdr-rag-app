# gemini-pdr-rag-app
🤖 Advanced Gemini RAG Chatbot (Parent Document Retriever)
This project demonstrates a fully functional, conversational Retrieval-Augmented Generation (RAG) system built with the LangChain framework and the Google Gemini API. It allows a user to upload a PDF document and chat with its content using multi-turn, context-aware dialogue.

🌟 Key Technical Achievements
This application implements advanced RAG techniques to overcome the primary challenges of context fragmentation and low answer quality:

Feature	LangChain Component(s)	Value Added
Advanced Indexing	Parent Document Retriever (PDR)	Solves context fragmentation. Indexes small chunks (for search precision) but retrieves large chunks (for rich LLM context).
Conversational Memory	ConversationalRetrievalChain & ConversationBufferMemory	Enables multi-turn dialogue. The system remembers previous questions, allowing users to ask follow-up questions (e.g., "What year was that?").
Answer Quality Control	combine_docs_chain_kwargs (Custom Prompt)	Enforces strict rules for the LLM output: concise, factual answers and a clear refusal if the answer is not in the document.
Deployment	Streamlit	Provides an instant, live, interactive web interface for user demonstration.

🚀 Deployment and Setup
This application is designed for instant deployment on Streamlit Community Cloud via this GitHub repository.

Prerequisites
A GitHub Account.

A Gemini API Key (or Google Cloud Key with Gemini access).

Your repository must contain the file app.py and requirements.txt.

1. Project File Structure
Ensure your repository's root directory contains these essential files:

/your-repo-name/
├── app.py              # The full Streamlit/RAG application code.
├── requirements.txt    # List of all Python dependencies.
└── .streamlit/         
    └── secrets.toml    # Contains your secure GEMINI_API_KEY.
2. Configuration (.streamlit/secrets.toml)
You must provide your Gemini API Key securely via Streamlit Secrets.

In your Streamlit Cloud workspace, access the Secrets management area for your app.

Create the file .streamlit/secrets.toml and paste your key in this exact format:

Ini, TOML

# .streamlit/secrets.toml
GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
3. Dependencies (requirements.txt)
This file is crucial for the deployment environment to install all necessary packages, including the community extensions:

# requirements.txt
streamlit
langchain
langchain-core
langchain-community
langchain-google-genai
pypdf
chromadb
4. Deploy to Streamlit Cloud
Log in to Streamlit Cloud with your GitHub account.

Click "New app".

Select your repository, the main branch, and set the Main file path to app.py.

Click "Deploy!"

🧠 Testing and Verification
Use these scenarios to verify the advanced RAG features:

Scenario	Test Question	Verification Point
PDR/Fact Check	"Was a chess player mentioned in the document, and what was his name?"	The answer should be affirmed, proving the PDR successfully retrieved the large, surrounding context chunk containing the detail.
Conversation Memory	Turn 1: "Who was the UN Secretary General?" Turn 2: "What did he say about AI?"	The LLM must correctly answer the second question by looking up the quote, proving history rephrasing is active.
Quality Control	"Who won the World Series last year?"	The model must strictly adhere to the refusal rule: "I could not find that specific detail in the document."
