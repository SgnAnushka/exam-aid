import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Ohm's Law states that voltage equals current multiplied by resistance."

vector = model.encode(text).tolist()

client.upsert(
    collection_name="study_material",
    points=[
        {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": text,
                "topic": "ohms law"
            }
        }
    ]
)

print("✅ Seed data inserted")
