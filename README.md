# ExamAid

A RAG-based chemistry study assistant that retrieves relevant compound information and generates accurate, context-aware explanations using LLMs.

---

## Features

- **Semantic Search** — Natural language queries for chemistry concepts
- **RAG Answers** — Responses grounded in curated knowledge base
- **Image Retrieval** — Structural diagrams of chemical compounds
- **Smart Recommendations** — Suggests related topics based on query history
- **Conversation Memory** — Remembers previous queries  for context

---

## Tech Stack

- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB:** Qdrant
- **Language:** Python 3.10+

---

## Architecture

```
          User Query
             ↓
      Query Processing
             ↓
    Text Retrieval (Qdrant)
             ↓
   Entity Grounding (Compound)
             ↓
    Image Retrieval (Qdrant)
             ↓
   Image–Text Intersection
             ↓
 RAG Answer Generation (LLM)
             ↓
    Recommendation Engine
             ↓
      Final Response
    └─────────────────┘
```

---

## Setup

```bash
# Clone repository
git clone https://github.com/SgnAnushka/exam-aid.git
cd exam-aid

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/macOS

# Install dependencies
pip install sentence-transformers qdrant-client groq python-dotenv pandas

# Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Run
python query-testing.py
```

---
