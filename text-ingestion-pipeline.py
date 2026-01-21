import csv
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = "dataset/images/test/chemistry3.csv"
COLLECTION_NAME = "study_text"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

# ----------------------------
# MODELS & DB
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333, timeout=60)

# ----------------------------
# CREATE COLLECTION
# ----------------------------
existing = [c.name for c in qdrant.get_collections().collections]

if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")
else:
    print(f"Collection exists: {COLLECTION_NAME}")

# ----------------------------
# INGEST
# ----------------------------
points = []
skipped = 0

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        compound_id = row.get("compound", "").strip()
        name = row.get("compoundLabel", "").strip()
        article = row.get("article", "").strip()

        # Hard guard
        if not name or not article:
            skipped += 1
            continue

        # Raw factual content ONLY
        content = article

        embedding = embedder.encode(content).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "type": "text",
                    "compound_id": compound_id,
                    "compound_name": name,
                    "content": content,
                    "source": "Wikidata"
                }
            )
        )

# ----------------------------
# UPSERT
# ----------------------------
if points:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"✅ Ingested {len(points)} text entries")
    print(f"⚠️ Skipped {skipped} invalid rows")
else:
    print("❌ No valid rows ingested")
