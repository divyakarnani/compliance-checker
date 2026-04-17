# EU Fashion Compliance Checker

A web app that analyzes fashion product marketing claims against EU regulations and flags compliance risks — built for customer discovery with fashion brands.

Upload a product catalog CSV, get instant risk reports with regulation citations and actionable fixes.

## Regulations Covered

- **ECGT** — EU Green Claims Directive (2024/825)
- **UCPD** — Unfair Commercial Practices Directive (2005/29/EC)
- **ESPR** — Ecodesign for Sustainable Products Regulation (2024/1781)

## How It Works

Each claim runs through a **five-gate rules engine** (deterministic + LLM hybrid):

| Gate | Method | What It Catches |
|------|--------|-----------------|
| 1. Blacklist | Python | Banned generic terms ("eco-friendly", "sustainable", etc.) |
| 2. Neutrality/Offsets | Python | Carbon neutral, climate positive, net zero claims |
| 3. Certification Match | Python | Validates cert scope (GOTS, GRS, Bluesign, FSC, TENCEL, etc.) |
| 4. Comparative Claims | Python + LLM | "X% less water" without third-party verification |
| 5. Ambiguous Cases | RAG + GPT-4o | Everything else — retrieves regulation context, then assesses |

Gates 1–3 are fully deterministic. Gates 4–5 use GPT-4o with RAG over the full regulation texts stored in ChromaDB.

## Features

- **Full catalog audit** — upload CSV, get per-product risk breakdown (HIGH / MEDIUM / LOW)
- **Conversational chat** — ask follow-up questions with streaming responses (SSE)
- **Regulation-grounded** — every finding cites specific articles from ECGT/UCPD/ESPR
- **Actionable fixes** — each flagged claim comes with a compliant rewrite suggestion

## Tech Stack

**Backend:** FastAPI, OpenAI (GPT-4o + text-embedding-3-small), ChromaDB, Pandas

**Frontend:** Vanilla HTML/CSS/JS (single file), SSE streaming

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser. On first startup, regulation texts are automatically ingested into ChromaDB.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Parse a product CSV |
| `/audit` | POST | Run compliance audit on products |
| `/chat` | POST | Conversational compliance chat (streaming) |
| `/health` | GET | Health check |

## Evaluation

```bash
python -m backend.eval
```

Runs 50 labeled test cases from `ground_truth_eval_dataset.csv` (sourced from EU Commission Q&A, Nov 2025) and scores on risk level, regulation citation, and issue identification accuracy.

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI server + routes
│   ├── analyze.py       # Five-gate rules engine
│   ├── ingest.py        # Regulation text → ChromaDB ingestion
│   ├── eval.py          # Regression test runner
│   └── regulations/     # ECGT, UCPD, ESPR full texts
├── frontend/
│   └── index.html       # Complete UI
├── sample_products.csv
├── ground_truth_eval_dataset.csv
└── requirements.txt
```
