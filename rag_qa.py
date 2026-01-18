import os
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from google import genai

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --------------------------------------------------
# Constants
# --------------------------------------------------
COLLECTION_NAME = "study_material"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embedding size

# --------------------------------------------------
# Initialize models (load once)
# --------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# --------------------------------------------------
# Initialize Qdrant
# --------------------------------------------------
qdrant = QdrantClient(host="localhost", port=6333)

# --------------------------------------------------
# Ensure collection exists (AUTO-INIT)
# --------------------------------------------------
existing_collections = [
    c.name for c in qdrant.get_collections().collections
]

if COLLECTION_NAME not in existing_collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Qdrant collection created: {COLLECTION_NAME}")
else:
    print(f"ℹ️ Qdrant collection exists: {COLLECTION_NAME}")

# --------------------------------------------------
# RAG function (used by FastAPI)
# --------------------------------------------------
def answer_question(question: str) -> str:
    """
    Retrieves relevant content from Qdrant and generates
    a grounded answer using Gemini.
    """

    # ---- Step 1: Embed the question ----
    query_vector = embed_model.encode(question).tolist()

    # ---- Step 2: Retrieve from Qdrant ----
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
        )
    except Exception:
        return "Knowledge base is currently unavailable. Please try again later."

    if not results.points:
        return (
            "I don’t have relevant material for this question yet. "
            "Please try another topic."
        )

    context = "\n".join(
        point.payload.get("text", "")
        for point in results.points
    )

    # ---- Step 3: Grounded prompt ----
    prompt = f"""
You are a study assistant.

Answer the question using ONLY the study material below.
If the answer is not present, say you don't know.
Do not add external information.

Study material:
{context}

Question:
{question}
"""

    # ---- Step 4: Generate answer ----
    response = gemini.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt,
    )

    return response.text


# to make venv (venv\Scripts\Activate.ps1)
# run docker (docker run -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant)
#start frontend (uvicorn app:app --reload)