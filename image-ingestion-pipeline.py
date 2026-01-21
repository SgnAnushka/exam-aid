import csv
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "dataset/images/test/query.csv"
COLLECTION_NAME = "study_images"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors

# ----------------------------
# Models & DB
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

# ----------------------------
# Create collection if it doesn't exist
# ----------------------------
collections = qdrant.get_collections().collections
if not any(c.name == COLLECTION_NAME for c in collections):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME}")

points = []

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        compound_id = row["compound"]
        compound_name = row["compoundLabel"]
        image_url = row["image"]

        # Honest, minimal text representation
        content = (
            f"Chemical compound: {compound_name}. "
            f"This entry corresponds to the structural image of the compound."
        )

        embedding = embedder.encode(content).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "type":"image",
                    "content": content,
                    "compound_id": compound_id,
                    "compound_name": compound_name,
                    "image_path": image_url,
                    "source": "Wikidata",
                }
            )
        )

if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} compound images")
else:
    print("No rows ingested")
