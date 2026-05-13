from langchain_huggingface import HuggingFaceEmbeddings
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup & Authentication
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY. Please check your .env file.")
    st.stop()

st.set_page_config(page_title="PDF Chatter", page_icon="📄", layout="wide")

# 2. Sidebar for Multi-File Upload
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    process_btn = st.button("Process PDFs", type="primary")

# 3. Dynamic Data Ingestion Engine
def process_documents(files):
    all_pages = []
    
    # Temporarily save each file to disk so LangChain can read it
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            
        # Load and append the pages
        loader = PyPDFLoader(temp_path)
        all_pages.extend(loader.load())
        
        # Clean up the temp file
        os.remove(temp_path)
        
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)
    
    # Create the Vector Database in memory
    # This downloads a tiny, lightning-fast open-source model to your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    return vector_db

# 4. Process the files when the user clicks the button
if process_btn and uploaded_files:
    with st.spinner("Reading and memorizing your documents..."):
        # Process the files and store the database in Streamlit's session state
        st.session_state["vector_db"] = process_documents(uploaded_files)
        st.success("✅ Documents processed! You can now chat.")

# 5. Main Chat Interface
st.title("📄 Chat with Multiple PDFs")

# Only show the chat box IF the database exists in the session state
if "vector_db" in st.session_state:
    db = st.session_state["vector_db"]
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Fetch top 5 chunks
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    system_prompt = (
        "You are an intelligent assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If the answer is not in the context, say "
        "'I cannot find this information in the provided documents.' Do not guess.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_input = st.text_input("Ask a question about your uploaded documents:")

    if user_input:
        with st.spinner("Searching..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                st.write("### Answer")
                st.write(response["answer"])
                
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response["context"]):
                        # Shows the text AND the name of the specific PDF it came from!
                        source_file = doc.metadata.get('source', 'Unknown')
                        st.write(f"**Source {i+1}:** (From {source_file})")
                        st.write(doc.page_content)
                        st.divider()
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("👈 Please upload and process some PDFs in the sidebar to get started.")