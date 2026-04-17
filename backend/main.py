"""FastAPI backend for EU Fashion Compliance checker."""

import asyncio
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from openai import OpenAI

from ingest import ensure_ingested
from analyze import (
    build_chat_context, CHAT_SYSTEM_PROMPT,
    analyze_claim, _parse_claims, _parse_certifications,
)

_executor = ThreadPoolExecutor(max_workers=8)

app = FastAPI(title="EU Fashion Compliance Checker")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

llm = OpenAI()


@app.on_event("startup")
async def startup():
    ensure_ingested()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Parse uploaded CSV and return product list as JSON."""
    content = await file.read()
    text = content.decode("utf-8")
    df = pd.read_csv(StringIO(text))

    # Normalize column names to lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    products = df.to_dict(orient="records")
    return {"products": products, "count": len(products)}


@app.post("/audit")
async def audit(request: Request):
    """Run full compliance audit on all products. Returns structured results."""
    body = await request.json()
    products = body.get("products", [])

    loop = asyncio.get_event_loop()

    # Build list of all (product_index, claim, certs, name) to analyze
    tasks = []
    for i, p in enumerate(products):
        name = p.get("name") or p.get("product_name") or f"Product {i+1}"
        claims_text = p.get("claims") or p.get("marketing_claims") or ""
        certs_text = p.get("certifications") or ""
        claims = _parse_claims(claims_text)
        certs = _parse_certifications(certs_text)
        for claim_str in claims:
            tasks.append((i, claim_str, certs, name))

    # Run all claim analyses concurrently via thread pool
    futures = [
        loop.run_in_executor(_executor, analyze_claim, claim_str, certs, name)
        for (_, claim_str, certs, name) in tasks
    ]
    claim_results_flat = await asyncio.gather(*futures)

    # Group results by product index
    product_claims: dict[int, list] = {}
    for (i, _, _, _), result in zip(tasks, claim_results_flat):
        product_claims.setdefault(i, []).append(result)

    results = []
    summary = {"total": 0, "high": 0, "medium": 0, "low": 0}

    for i, p in enumerate(products):
        name = p.get("name") or p.get("product_name") or f"Product {i+1}"
        claims_text = p.get("claims") or p.get("marketing_claims") or ""
        certs_text = p.get("certifications") or ""
        claim_results = product_claims.get(i, [])

        product_risk = "LOW"
        for result in claim_results:
            risk = result.get("overall_risk", "LOW")
            if risk == "HIGH":
                product_risk = "HIGH"
            elif risk == "MEDIUM" and product_risk != "HIGH":
                product_risk = "MEDIUM"

        results.append({
            "product_index": i,
            "product_name": name,
            "claims_raw": claims_text,
            "certifications_raw": certs_text,
            "overall_risk": product_risk,
            "claim_results": claim_results,
        })

        summary["total"] += 1
        if product_risk == "HIGH":
            summary["high"] += 1
        elif product_risk == "MEDIUM":
            summary["medium"] += 1
        else:
            summary["low"] += 1

    return {"results": results, "summary": summary}


@app.post("/chat")
async def chat(request: Request):
    """Conversational compliance chat. Streams GPT-4o response via SSE."""
    body = await request.json()
    messages = body.get("messages", [])
    products = body.get("products", [])

    # Extract latest user message for RAG retrieval
    latest_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_msg = msg.get("content", "")
            break

    # Build regulation context from balanced retrieval
    context = build_chat_context(latest_msg, products)

    # System prompt with regulation context appended
    system_content = CHAT_SYSTEM_PROMPT + context

    # Build LLM messages: system + full conversation history
    llm_messages = [{"role": "system", "content": system_content}]
    llm_messages.extend(messages)

    async def event_stream():
        try:
            stream = llm.chat.completions.create(
                model="gpt-4o",
                messages=llm_messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serve frontend static files — must be last to not override API routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
