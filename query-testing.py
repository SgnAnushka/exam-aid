import os
from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TEXT_COLLECTION = "study_text"
IMAGE_COLLECTION = "study_images"

TOP_K_TEXT = 1
TOP_K_IMAGE = 1

TEXT_SCORE_THRESHOLD = 0.25
IMAGE_SCORE_THRESHOLD = 0.15

# -------------------------------------------------
# INIT
# -------------------------------------------------
load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(host="localhost", port=6333, timeout=60)

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------
# TEXT RETRIEVAL
# -------------------------------------------------
def retrieve_text(query: str):
    query_vector = embedder.encode(query).tolist()

    hits = qdrant.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        limit=TOP_K_TEXT
    ).points

    texts = []
    sources = []

    for h in hits:
        if h.score < TEXT_SCORE_THRESHOLD:
            continue

        payload = h.payload or {}

        if payload.get("content"):
            texts.append(payload["content"])

            sources.append({
                "compound": payload.get("compound_name"),
                "score": round(h.score, 3)
            })

    return texts, sources


# -------------------------------------------------
# IMAGE RETRIEVAL
# -------------------------------------------------
def retrieve_images(query: str):
    query_vector = embedder.encode(query).tolist()

    hits = qdrant.query_points(
        collection_name=IMAGE_COLLECTION,
        query=query_vector,
        limit=TOP_K_IMAGE
    ).points

    images = []

    for h in hits:
        if h.score < IMAGE_SCORE_THRESHOLD:
            continue

        payload = h.payload or {}

        if payload.get("image_path"):
            images.append({
                "compound": payload.get("compound_name"),
                "image_url": payload["image_path"],
                "score": round(h.score, 3)
            })

    return images


# -------------------------------------------------
# RAG ANSWER
# -------------------------------------------------
def answer_query(query: str):
    texts, sources = retrieve_text(query)
    images = retrieve_images(query)

    if not texts and not images:
        return {
            "answer": "❌ No relevant data found in the knowledge base.",
            "images": [],
            "sources": []
        }

    if texts:
        context_text = "\n\n".join(texts)

        prompt = f"""
You are an academic chemistry tutor.

TASK:
- Read the provided article content
- Write a clear, structured explanation (10–15 lines)
- Use ONLY the provided content
- If information is missing, explicitly say so

ARTICLE CONTENT:
{context_text}

QUESTION:
{query}
"""

        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()

    else:
        answer = "ℹ️ Relevant images are available, but no textual explanation is stored."

    return {
        "answer": answer,
        "images": images,
        "sources": sources
    }


# -------------------------------------------------
# CLI TEST
# -------------------------------------------------
if __name__ == "__main__":
    query = input("Ask a question: ").strip()

    result = answer_query(query)

    print("\n================ ANSWER ================\n")
    print(result["answer"])

    print("\n================ IMAGES ================\n")
    if result["images"]:
        for img in result["images"]:
            print(img)
    else:
        print("No images available.")

    print("\n================ SOURCES ===============\n")
    for s in result["sources"]:
        print(s)