
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

DB_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"
MODEL_NAME = "qwen2.5:3b"

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 AI Research Assistant")
st.write("Ask questions about your PDF documents.")


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedding_model


try:
    collection, embedding_model = load_resources()
except Exception as e:
    st.error(f"Could not load the document database: {e}")
    st.info("Check that you have run `python ingest.py` successfully.")
    st.stop()


question = st.text_input(
    "Enter your question",
    placeholder="What are CNNs commonly used for?",
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Searching documents and generating an answer..."):
        try:
            question_embedding = embedding_model.encode(
                question,
                normalize_embeddings=True,
            ).tolist()

            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3,
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            if not documents:
                st.warning("No relevant information was found.")
                st.stop()

            context_parts = []

            for document, metadata in zip(documents, metadatas):
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")

                context_parts.append(
                    f"Source: {source}, Page: {page}\n{document}"
                )

            context = "\n\n".join(context_parts)

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using only the supplied "
                            "document context. If the answer is not present, "
                            "say you could not find it in the documents. "
                            "Do not invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            )

            answer = response["message"]["content"]

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")

            unique_sources = set()

            for metadata in metadatas:
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")
                unique_sources.add((source, page))

            for source, page in sorted(unique_sources):
                st.write(f"- {source}, Page {page}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
