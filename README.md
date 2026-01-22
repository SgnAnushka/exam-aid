# ExamAid 🧪  
**A Multimodal RAG-based Study Assistant**

ExamAid is a Retrieval-Augmented Generation (RAG) system designed to help students understand  concepts accurately and contextually. It combines **semantic text retrieval**, **compound-aware image retrieval**, and **LLM-based reasoning**, all grounded using a **Qdrant vector database**.

---

## 🚀 Problem Statement

students often struggle with:
- Abstract explanations disconnected from real compounds
- Hallucinated or inaccurate AI-generated answers
- Lack of visual grounding (structures, diagrams)
- No continuity across multiple study queries

**ExamAid** solves this by grounding every answer in a curated knowledge base using vector search and controlled generation.

---

## ✨ Features

- **Semantic Text Search**  
  Ask questions in natural language.

- **Grounded RAG Answers**  
  Answers are generated *only* from retrieved database content.

- **Compound-Based Image Retrieval**  
  Structural images are retrieved using vector similarity, not keywords.

- **Smart Recommendations**  
  Suggests related compounds/topics using similarity search.

- **Conversation Memory**  
  Maintains context without polluting retrieval.

---

## 🧠 Tech Stack

- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Text Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database:** Qdrant
- **Language:** Python 3.10+
- **Data:** Curated articles + compound structure images

---

## 🏗️ System Architecture (Unified Pipeline)
┌──────────────────────────────────────────┐
│ USER │
│ Natural Language Query │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ QUERY PROCESSING │
│ - cleaning │
│ - lowercasing │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ SESSION MEMORY │
│ - stores last N (≤5) queries │
│ - avoids duplicates │
│ - NOT used in vector search │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────── QDRANT ─────────────────────────┐
│ │
│ ┌───────────────────────────────────────────────────┐ │
│ │ TEXT_COLLECTION │ │
│ │ │ │
│ │ chemistry3.csv │ │
│ │ ↓ SentenceTransformer (384-d) │ │
│ │ ↓ Stored as TEXT VECTORS │ │
│ │ │ │
│ │ Query Vector → Semantic Search │ │
│ │ ↓ │ │
│ │ Top-K Text Hits (Compound + Description) │ │
│ └───────────────────────────────────────────────────┘ │
│ │
│ ▲ │
│ │ │
│ Resolve abstract query │
│ → Representative compound │
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────┐ │
│ │ IMAGE_COLLECTION │ │
│ │ │ │
│ │ query.csv │ │
│ │ ↓ "chemical structure of X" │ │
│ │ ↓ SentenceTransformer (384-d) │ │
│ │ ↓ Stored as IMAGE VECTORS │ │
│ │ │ │
│ │ Compound-based Query Vector → Search │ │
│ │ ↓ │ │
│ │ Top-K Image Matches (IDs + similarity scores) │ │
│ │ (Image URLs stored as metadata only) │ │
│ └───────────────────────────────────────────────────┘ │
│ │
└──────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ IMAGE–TEXT INTERSECTION │
│ - verifies text exists for image │
│ - removes orphan image results │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ BEST COMPOUND SELECTION │
│ Score = image_score + text_score │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ RAG PROMPT CONSTRUCTION │
│ - compound description │
│ - user memory │
│ - strict grounding instruction │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ LLM (GROQ) │
│ - generates grounded explanation │
│ - no hallucination │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ RECOMMENDATION ENGINE │
│ - similarity search in TEXT_COLLECTION │
│ - suggests related compounds │
└─────────────────────┬────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│ FINAL RESPONSE │
│ - Answer │
│ - Recommended topics │
│ - Image matches + scores │
│ - Memory context │
└──────────────────────────────────────────┘


---

## 🧩 Why Qdrant Is Critical

- Stores **both text and image embeddings**
- Enables **fast semantic similarity search**
- Keeps retrieval **deterministic and explainable**
- Prevents hallucination by controlling context
- Acts as the **single source of truth** for RAG

---

## ⚠️ Limitations & Ethics

- Answers are limited to dataset coverage
- No personal data stored
- No medical or hazardous advice generated

---

## 🛠️ Setup Instructions

```bash
# Clone repository
git clone https://github.com/SgnAnushka/exam-aid.git
cd exam-aid

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install sentence-transformers qdrant-client groq python-dotenv pandas

# Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Run the system
python query-testing.py