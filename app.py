import os
import pickle
import sys
import streamlit as st
from dotenv import load_dotenv

# --- 1. IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever

load_dotenv()

# --- 2. CONFIGURATION ---
INDEX_NAME = "branham-index"
CHUNKS_FILE = "sermon_chunks.pkl"

def get_rag_chain():
    # --- A. AUTHENTICATION (Hugging Face Fix) ---
    # We MUST check os.environ first to avoid the "No secrets found" crash on HF
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    # Only look for local Streamlit secrets if Env vars are missing
    if not pinecone_key or not google_key:
        try:
            if not pinecone_key:
                pinecone_key = st.secrets.get("PINECONE_API_KEY")
            if not google_key:
                google_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            # Squelch the error so HF doesn't crash
            pass

    if not pinecone_key or not google_key:
        raise ValueError("❌ Missing API Keys. Go to Settings > Variables and secrets on Hugging Face and add PINECONE_API_KEY and GOOGLE_API_KEY.")

    # Set globally for LangChain
    os.environ["PINECONE_API_KEY"] = pinecone_key
    os.environ["GOOGLE_API_KEY"] = google_key

    # --- B. PINECONE (Cloud Vector DB) ---
    print("🔌 Connecting to Pinecone...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    # Increased context window to 8 paragraphs
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # --- C. KEYWORD SEARCH (Local File) ---
    print("🔌 Loading Keyword Search...")
    keyword_retriever = None
    
    try:
        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = 8 # Match context size
        else:
            print(f"⚠️ {CHUNKS_FILE} missing. Using Vector only.")
    except Exception as e:
        print(f"❌ Failed to load keyword file: {e}")

    # --- D. HYBRID MERGE ---
    if keyword_retriever:
        print("🔗 Linking Hybrid System...")
        final_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
    else:
        final_retriever = vector_retriever

    # --- E. MODEL (Pro Version) ---
    # Note: Using 1.5 Pro as the stable high-intelligence model.
    # If 2.5-pro is a valid Preview model you have access to, change string below.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.3,
        convert_system_message_to_human=True
    )

    # --- F. PERSONA PROMPT ---
    template = """You are William Marion Branham.
INSTRUCTIONS:
- You are a helpful evangelist. Answer the question comprehensively using the Context below.
- Speak in the first person ("I said," "The Lord showed me").
- Use a humble, 1950s Southern preaching dialect.
- If the Context mentions the topic but isn't complete, use your general knowledge of the Message to fill in the gaps, but prioritize the text provided.
- Do not say "I don't recall" unless the topic is completely unrelated to the Bible or the Message.
CONTEXT:
{context}
USER QUESTION: {question}
BROTHER BRANHAM'S REPLY:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain
