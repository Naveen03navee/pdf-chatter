# PDF Chatter: Multi-Document RAG Application

A full-stack, containerized AI platform that allows users to upload multiple PDF documents and engage in a contextual conversation. The system uses a Retrieval-Augmented Generation (RAG) architecture.

## Architecture Overview
- Frontend: React (Vite) with a custom chat interface.
- Backend: FastAPI REST API for document processing and LLM orchestration.
- AI Logic: LangChain framework.
- Embeddings: Local HuggingFace sentence-transformers (all-MiniLM-L6-v2).
- LLM: Google Gemini 2.5 Flash.
- Database: ChromaDB (Vector Store).
- DevOps: Docker Compose for multi-container orchestration.

## Features
- Local Embedding Generation: No API costs or rate limits for document indexing.
- Multi-PDF Support: Process and query across multiple documents simultaneously.
- Containerized Environment: Entire stack runs in isolated Docker containers.
- Markdown Rendering: Professional formatting for AI-generated responses.

## Installation and Setup

### Prerequisites
- Docker and Docker Compose
- Google Gemini API Key

### Configuration
1. Clone the repository.
2. Create a .env file in the root directory.
3. Add your API key:
   GEMINI_API_KEY=your_api_key_here

### Running the Application
Build and start the services using Docker Compose:

docker-compose up --build

Once the build is complete:
- Access the React Frontend: http://localhost:5173
- Access the FastAPI Documentation: http://localhost:8000/docs