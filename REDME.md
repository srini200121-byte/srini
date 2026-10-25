# AI Research Assistant — RAG-Based PDF Question Answering

## Project Overview
An AI-powered research assistant that answers questions using information retrieved from PDF documents. It uses Retrieval-Augmented Generation (RAG) to provide answers with document source references.

## Features
- Extract text from PDF documents
- Split documents into chunks
- Generate text embeddings
- Store and retrieve chunks using ChromaDB
- Generate answers using a local LLM through Ollama
- Display source filenames and page numbers
- Streamlit web interface

## Tech Stack
- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- PyMuPDF
- Ollama
- Qwen2.5:3b

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Install Ollama and download the model:

   `ollama pull qwen2.5:3b`

5. Add your PDF documents to the `data/` directory.
6. Build the document database:

   `python ingest.py`

7. Start the application:

   `python -m streamlit run app.py`

## How It Works
1. PDF documents are extracted and split into chunks.
2. Sentence Transformers converts chunks into embeddings.
3. ChromaDB retrieves relevant chunks for each question.
4. Ollama generates an answer using the retrieved context.
5. The application displays the answer and its sources.

## Future Improvements
- Support multiple PDF documents reliably
- Improve retrieval quality and answer evaluation
- Add document upload through the web interface
- Add automated tests

## Author
AI/ML and RAG Engineering Portfolio Project