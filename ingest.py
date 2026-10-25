
from pathlib import Path
import hashlib

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)


def load_and_chunk():
    records = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        if not pdf_path.is_file():
            print(f"Skipping  folder:{pdf_path}")
            continue
        with pymupdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                words = text.split()

                chunk_size = 500
                overlap = 80
                step = chunk_size - overlap

                for start in range(0, len(words), step):
                    chunk = " ".join(
                        words[start:start + chunk_size]
                    )

                    if chunk:
                        records.append({
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_num,
                        })

                    if start + chunk_size >= len(words):
                        break

    return records


def main():
    records = load_and_chunk()

    if not records:
        print("No PDF text found. Check your data folder.")
        return

    # Rebuild the collection when ingesting again.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    # Add records in batches.
    batch_size = 64

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [item["text"] for item in batch]

        embeddings = model.encode(
            texts, normalize_embeddings=True
        ).tolist()

        ids = [
            hashlib.sha256(
                f"{item['source']}|{item['page']}|{start+i}|"
                f"{item['text']}".encode()
            ).hexdigest()
            for i, item in enumerate(batch)
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": item["source"],
                    "page": item["page"],
                }
                for item in batch
            ],
        )

    print(f"Indexed {len(records)} chunks successfully.")


if __name__ == "__main__":
    main()
