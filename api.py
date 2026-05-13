import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(title="PDF Chatter API")

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store our in-memory database
vector_db = None

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
        
    try:
        # Load and split the PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        # Create local embeddings and store in memory
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        return {"message": f"Successfully processed {file.filename}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    global vector_db
    
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
        
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        system_prompt = (
            "You are an intelligent assistant. Use the following pieces of retrieved context "
            "to answer the user's question. If the answer is not in the context, say "
            "'I cannot find this information in the provided document.'\n\nContext:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": request.message})
        
        return {"answer": response["answer"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)