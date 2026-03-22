"""
Vera Verifier Server - ModernBERT Zeroshot
Uses MoritzLaurer/ModernBERT-large-zeroshot-v2.0 for claim verification
Binary classification: entailment (SUPPORTED) vs not_entailment (REFUTED)
"""

import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

app = FastAPI(title="Vera Verifier", version="2.0")

MODEL_ID = os.getenv("MODEL_ID", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0")
MODEL_CACHE = "/app/models"
PORT = int(os.getenv("PORT", 8081))

print(f"Loading model: {MODEL_ID}")
classifier = pipeline(
    "zero-shot-classification", 
    model=MODEL_ID, 
    device=-1,  # CPU, use 0 for GPU
    model_kwargs={"cache_dir": MODEL_CACHE}
)
print("Model loaded!")

# Binary labels for NLI classification (matches CoT verifier)
CANDIDATE_LABELS = ["entailment", "not_entailment"]
VERDICT_MAP = {
    "entailment": "SUPPORTED",
    "not_entailment": "REFUTED"
}


class VerifyRequest(BaseModel):
    claim: str
    evidence: str


class VerifyBatchRequest(BaseModel):
    claims: List[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/verify")
def verify_claim(req: VerifyRequest):
    """Verify a claim against evidence using NLI (binary classification)."""
    # Combine evidence and claim for classification
    sequence = f"Evidence: {req.evidence}\n\nClaim: {req.claim}"
    
    result = classifier(
        sequence, 
        CANDIDATE_LABELS, 
        hypothesis_template="This claim is {}."
    )
    
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    
    return {
        "claim": req.claim,
        "evidence": req.evidence,
        "verdict": VERDICT_MAP[top_label],
        "confidence": top_score,
        "raw_label": top_label,
        "all_scores": dict(zip(result["labels"], result["scores"]))
    }


@app.post("/verify_batch")
def verify_batch(req: VerifyBatchRequest):
    """Verify multiple claims at once."""
    results = []
    
    for item in req.claims:
        claim = item.get("claim", "")
        evidence = item.get("evidence", "")
        
        if not claim or not evidence:
            results.append({
                "claim": claim,
                "evidence": evidence,
                "verdict": "ERROR",
                "confidence": 0,
                "raw_label": "MISSING_INPUT"
            })
            continue
        
        sequence = f"Evidence: {evidence}\n\nClaim: {claim}"
        
        result = classifier(
            sequence, 
            CANDIDATE_LABELS, 
            hypothesis_template="This claim is {}."
        )
        
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        results.append({
            "claim": claim,
            "evidence": evidence,
            "verdict": VERDICT_MAP[top_label],
            "confidence": top_score,
            "raw_label": top_label
        })
    
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
