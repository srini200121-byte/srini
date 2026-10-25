
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

DB_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"
MODEL_NAME = "qwen2.5:3b"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def answer_question(question):
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

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
                    "Answer using only the supplied context. "
                    "If the answer is missing, say you could not "
                    "find it in the provided documents. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    print("\n--- RAG Answer ---")
    print(response["message"]["content"])

    print("\n--- Sources ---")
    for metadata in metadatas:
        print(
            f"File: {metadata.get('source', 'Unknown')}, "
            f"Page: {metadata.get('page', 'Unknown')}"
        )


if __name__ == "__main__":
    question = input("Enter a question about your PDFs: ")
    answer_question(question)
