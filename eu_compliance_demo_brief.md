# EU Fashion Compliance Demo — Project Brief

## What We're Building

A web app that accepts a fashion brand's product catalog as input, runs each product against real EU regulation text using RAG, and outputs a risk report showing which products have compliance issues, why, and what to fix.

This is a demo tool for customer discovery calls with fashion brand sustainability leads. The goal is to show it live on a call, get them to say "I want to see my actual products in here", and use that as the wedge to get real product data.

---

## Tech Stack

- **Backend** — Python, FastAPI
- **Frontend** — Single HTML page, no framework
- **RAG** — LangChain + ChromaDB
- **Embeddings** — OpenAI text-embedding-3-small
- **LLM** — GPT-4o
- **File handling** — pandas for CSV parsing

---

## Project Structure

```
compliance-demo/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── ingest.py            # Regulation document ingestion
│   ├── analyze.py           # RAG analysis logic
│   └── regulations/         # Raw regulation text files
│       ├── ecgt.txt
│       ├── ucpd.txt
│       └── espr.txt
├── frontend/
│   └── index.html           # Single page UI
├── sample_data/             # Drop real product CSVs here
├── .env                     # API keys
└── requirements.txt
```

---

## Regulation Documents

Fetch the actual text from EUR-Lex and save to the regulations/ folder before running:

- **ECGT** — https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024L0825
- **UCPD** — https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32005L0029
- **ESPR** — https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1781

---

## Step 1 — Regulation Ingestion (ingest.py)

Load the three regulation documents from the regulations/ folder. **Chunk by article**, not by fixed token size — each chunk should be one article or sub-article.

Each chunk must store this metadata:
- `regulation_name` — e.g. "ECGT"
- `article_number` — e.g. "Article 6(2)"
- `article_title`
- `full_text`

Embed all chunks using text-embedding-3-small and store in ChromaDB. Run once on startup if the vector store doesn't exist yet.

---

## Step 2 — Product Analysis (analyze.py)

For each product row in the uploaded CSV, run this pipeline:

**Query construction** — build a query from the product's marketing claims and material composition:
```
"compliance requirements for [claim] under EU fashion sustainability regulations"
```

**Retrieval** — fetch top 5 most relevant regulation chunks from ChromaDB.

**Analysis** — send to GPT-4o with this system prompt:

```
You are an EU fashion compliance expert. Given a product's details 
and relevant EU regulation text, identify specific compliance risks.

For each risk you identify you must:
- Name the exact regulation and article number
- Quote the specific requirement being violated
- Explain in plain English what the brand needs to do to fix it
- Rate the risk as HIGH, MEDIUM, or LOW

If no compliance risk exists, say so clearly.

Never invent regulations. Only cite what is in the provided context.
If the context doesn't contain enough information to make a 
determination, say so explicitly.
```

**Output format per product:**

```json
{
  "product_name": "",
  "claims_analyzed": [],
  "risks": [
    {
      "risk_level": "HIGH",
      "claim": "",
      "regulation": "",
      "article": "",
      "requirement": "",
      "fix": ""
    }
  ],
  "overall_risk": "HIGH/MEDIUM/LOW/COMPLIANT"
}
```

---

## Step 3 — FastAPI Backend (main.py)

Three endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Accept CSV, parse with pandas, return product list for confirmation |
| `/analyze` | POST | Run RAG analysis, stream results back as each product completes |
| `/health` | GET | Health check |

Stream results back as they complete — do not make the user wait for the full batch.

---

## Step 4 — Frontend (index.html)

Single page, clean and professional. Three states:

**State 1 — Upload**
- CSV drag and drop upload area
- Single claim text input for quick demos without a CSV — this is the most important feature
- "Run Analysis" button

**State 2 — Loading**
- Progress indicator showing which product is currently being analyzed
- Results stream in as they complete

**State 3 — Results**
- Product name and claims analyzed
- Overall risk badge — RED for HIGH, YELLOW for MEDIUM, GREEN for COMPLIANT
- Expandable section per risk showing regulation, article number, issue, and fix
- "Download Report" button — exports full results as PDF

**Design requirements:**
- Clean and minimal — looks like a real product not a hackathon project
- Mobile responsive
- No external CSS frameworks

---

## Product Data

**No made-up data.** Real product data only.

### What to look for

Find a CSV or manually pull product data from brand websites with these columns:

| Column | Example |
|---|---|
| `product_name` | Organic Cotton Crewneck |
| `material_composition` | 100% organic cotton |
| `marketing_claims` | carbon neutral, sustainably made |
| `certifications` | GOTS, Fair Trade |
| `country_of_manufacture` | Portugal |
| `price_point` | 120 |

### Where to find real data

- **Reformation** — product pages list material composition and sustainability claims explicitly, easy to copy 10-15 products manually
- **Patagonia** — very detailed product-level environmental claims
- **Veja** — good for certifications
- **Allbirds** — carbon footprint listed per product
- **Asket** — lists full material breakdown and impact per garment

Pull 10-15 products manually from one brand. That becomes your demo dataset and makes the demo feel real when you're on a call with that brand specifically.

---

## Critical Requirements

- **Every risk flagged must cite a real article** from the ingested regulation documents — if RAG retrieval doesn't find a relevant article, output "insufficient information", never hallucinate a citation
- **Stream results to frontend as they complete** — do not batch wait
- **Single claim text input is the killer demo feature** — during a call type "climate positive" and show analysis in 10 seconds without needing a CSV
- **No sample/fake product data** — leave the sample_data/ folder empty, real data only

---

## Environment Variables

```
OPENAI_API_KEY=sk-proj-your-key-here
```

---

## The Demo Moment

During a customer discovery call, share your screen and type a single claim — "climate positive", "carbon neutral", "sustainably made" — into the text input. Show the risk analysis come back in real time citing the actual regulation and article. That's the moment that makes them say "I want to see my actual products in here."

That ask — "would you be willing to share 10-20 products so I can run this against your actual claims?" — becomes a much easier yes after they've seen it work.
