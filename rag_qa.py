import os
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai

# ---------- Load environment ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---------- Models ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- Qdrant ----------
qdrant = QdrantClient(host="localhost", port=6333)

# ---------- Student question ----------
question = "Explain what is a butterfly."

# ---------- Step 1: Embed the question ----------
query_vector = embed_model.encode(question).tolist()

# ---------- Step 2: Retrieve study content from Qdrant ----------
results = qdrant.query_points(
    collection_name="study_material",
    prefetch=[],
    query=query_vector,
    limit=2,
)

retrieved_texts = [p.payload["text"] for p in results.points]

context = "\n".join(retrieved_texts)

print("📚 Retrieved study content:\n")
print(context)

# ---------- Step 3: Ask Gemini using ONLY retrieved content ----------
prompt = f"""
You are a study assistant.
Answer the question using ONLY the study material below.
Do not add external information.

Study material:
{context}

Question:
{question}
"""

response = gemini.models.generate_content(
    model="models/gemini-flash-latest",
    contents=prompt,
)

print("\n🤖 Final Answer:\n")
print(response.text)
