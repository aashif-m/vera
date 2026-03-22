"""
Vera API - LangChain Pipeline Orchestrator
Connects decomposer, Tavily, and verifier
Supports both Standard and CoT (Chain-of-Thought) modes
"""

import os
import json
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from tavily import TavilyClient

app = FastAPI(title="Vera API", version="2.0")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Standard Mode
DECOMPOSER_URL = os.getenv("DECOMPOSER_URL", "http://decomposer:8080")
VERIFIER_URL = os.getenv("VERIFIER_URL", "http://verifier:8081")

# Configuration - CoT Mode
DECOMPOSER_COT_URL = os.getenv("DECOMPOSER_COT_URL", "http://decomposer-cot:8080")
VERIFIER_COT_URL = os.getenv("VERIFIER_COT_URL", "http://verifier-cot:8080")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ===========================================
# GBNF Grammars (loaded at startup)
# ===========================================

GRAMMAR_DIR = Path(__file__).parent / "grammars"

def _load_grammar(name: str) -> str | None:
    """Load a GBNF grammar file, return None if not found."""
    path = GRAMMAR_DIR / f"{name}.gbnf"
    if path.exists():
        return path.read_text(encoding="utf-8")
    print(f"WARNING: Grammar file not found: {path}")
    return None

GRAMMAR_DECOMP_STANDARD = _load_grammar("decomposer_standard")
GRAMMAR_DECOMP_COT = _load_grammar("decomposer_cot")
GRAMMAR_VERIFIER_COT = _load_grammar("verifier_cot")

# ===========================================
# System Prompts
# ===========================================

# Standard decomposer prompt (no reasoning)
DECOMPOSER_PROMPT_STANDARD = """You are a fact decomposer. Break text into atomic claims with exact quotes.

RULES:
1. Extract all claims - mark verifiable facts as FACTUAL, opinions as OPINION.
2. Each claim must stand alone (replace "he/she/it/they" with actual names).
3. Quote must be EXACT substring from input.
4. Return flat JSON array.

OUTPUT FORMAT:
[
  {
    "quote": "exact text from input",
    "atomic_claim": "Self-contained statement",
    "type": "FACTUAL"
  }
]

If no claims found, return: []"""

# CoT decomposer prompt (with reasoning)
DECOMPOSER_PROMPT_COT = """You are a fact decomposer. Break text into atomic claims with exact quotes.

RULES:
1. Extract all claims - mark verifiable facts as FACTUAL, opinions as OPINION.
2. Each claim must stand alone (replace "he/she/it/they" with actual names).
3. Quote must be EXACT substring from input.
4. Provide a brief reason for each type classification.
5. Return flat JSON array.

OUTPUT FORMAT:
[
  {
    "quote": "exact text from input",
    "atomic_claim": "Self-contained statement",
    "reason": "verifiable fact about X",
    "type": "FACTUAL"
  },
  {
    "quote": "opinion text from input",
    "atomic_claim": "Opinion statement",
    "reason": "subjective judgment using Y",
    "type": "OPINION"
  }
]

If no claims found, return: []"""

# CoT verifier prompt (LLM-based with reasoning)
VERIFIER_PROMPT_COT = """You are a fact-checker. Verify claims against evidence.

TASK: Given a CLAIM and EVIDENCE snippets, determine if the claim is supported or refuted.

OUTPUT FORMAT (JSON):
{
  "reasoning": "<analyze each evidence piece, then conclude>",
  "verdict": "SUPPORTED" | "REFUTED"
}

RULES:
1. SUPPORTED = claim confirmed by evidence
2. REFUTED = evidence contradicts the claim
3. Reference evidence by number: "Evidence 1 shows..."
4. If any key part is contradicted, verdict is REFUTED"""


# ===========================================
# Request/Response Models
# ===========================================

class CheckRequest(BaseModel):
    text: str
    use_reasoning: bool = False  # Toggle for CoT mode


class ClaimResult(BaseModel):
    quote: str
    atomic_claim: str
    claim_type: str
    reason: Optional[str] = None  # Only present in CoT mode
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    source: Optional[str] = None
    reasoning: Optional[str] = None  # Verifier reasoning (CoT mode only)


class CheckResponse(BaseModel):
    claims: List[ClaimResult]
    factuality_score: float
    supported_count: int
    refuted_count: int
    nei_count: int
    opinion_count: int
    mode: str  # "standard" or "reasoning"


# ===========================================
# Core Functions
# ===========================================

async def decompose_text(text: str, use_reasoning: bool = False) -> List[dict]:
    """Call decomposer to break text into claims."""
    
    # Select endpoint and prompt based on mode
    if use_reasoning:
        url = DECOMPOSER_COT_URL
        prompt = DECOMPOSER_PROMPT_COT
    else:
        url = DECOMPOSER_URL
        prompt = DECOMPOSER_PROMPT_STANDARD
    
    # Select grammar for constrained decoding
    grammar = GRAMMAR_DECOMP_COT if use_reasoning else GRAMMAR_DECOMP_STANDARD

    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if grammar:
            payload["grammar"] = grammar

        response = await client.post(
            f"{url}/v1/chat/completions",
            json=payload,
        )
        
        if response.status_code != 200:
            raise HTTPException(500, f"Decomposer error: {response.text}")
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        try:
            # Try to extract JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                claims = json.loads(content[start:end])
                return claims
        except json.JSONDecodeError:
            pass
        
        return []


async def retrieve_evidence(claim: str) -> tuple[str, str]:
    """Use Tavily to retrieve evidence for a claim."""
    if not TAVILY_API_KEY:
        return "No Tavily API key configured", ""
    
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        results = tavily.search(
            query=claim,
            search_depth="basic",
            max_results=3,
            include_answer=True
        )
        
        # Combine top results as evidence
        if results.get("answer"):
            evidence = results["answer"]
        else:
            evidence = " ".join([r.get("content", "")[:200] for r in results.get("results", [])])
        
        source = results.get("results", [{}])[0].get("url", "") if results.get("results") else ""
        
        return evidence[:500], source
    except Exception as e:
        return f"Evidence retrieval failed: {str(e)}", ""


async def verify_claim_standard(claim: str, evidence: str) -> tuple[str, float, str]:
    """Call ModernBERT verifier to check claim against evidence (Standard mode)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{VERIFIER_URL}/verify",
            json={"claim": claim, "evidence": evidence}
        )
        
        if response.status_code != 200:
            return "ERROR", 0.0, ""
        
        result = response.json()
        return result["verdict"], result["confidence"], ""


async def verify_claim_cot(claim: str, evidence: str) -> tuple[str, float, str]:
    """Call LLM verifier with reasoning (CoT mode)."""
    
    # Format the input for the verifier
    user_content = f"""CLAIM: {claim}

EVIDENCE:
{evidence}"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "messages": [
                {"role": "system", "content": VERIFIER_PROMPT_COT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        if GRAMMAR_VERIFIER_COT:
            payload["grammar"] = GRAMMAR_VERIFIER_COT

        response = await client.post(
            f"{VERIFIER_COT_URL}/v1/chat/completions",
            json=payload,
        )
        
        if response.status_code != 200:
            return "ERROR", 0.0, ""
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            # Try to extract JSON object
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                verdict = parsed.get("verdict", "NOT_ENOUGH_INFO")
                reasoning = parsed.get("reasoning", "")
                
                # Normalize verdict
                verdict = verdict.upper()
                if verdict not in ["SUPPORTED", "REFUTED"]:
                    verdict = "NOT_ENOUGH_INFO"
                
                # LLM doesn't provide confidence, use 0.8 as default
                return verdict, 0.8, reasoning
        except json.JSONDecodeError:
            pass
        
        return "NOT_ENOUGH_INFO", 0.5, ""


async def verify_claim(claim: str, evidence: str, use_reasoning: bool = False) -> tuple[str, float, str]:
    """Route to appropriate verifier based on mode."""
    if use_reasoning:
        return await verify_claim_cot(claim, evidence)
    else:
        verdict, confidence, _ = await verify_claim_standard(claim, evidence)
        return verdict, confidence, ""


def calculate_factuality_score(claims: List[ClaimResult]) -> float:
    """Calculate overall factuality score."""
    factual_claims = [c for c in claims if c.claim_type == "FACTUAL"]
    
    if not factual_claims:
        return 100.0  # No factual claims to verify
    
    score = 0
    for claim in factual_claims:
        if claim.verdict == "SUPPORTED":
            score += 1.0
        elif claim.verdict == "REFUTED":
            score += 0.0  # Refuted claims contribute 0% factuality
        else:
            # NOT_ENOUGH_INFO
            score += 0.5
            
    # Calculate simple average
    avg_score = score / len(factual_claims)
    return round(avg_score * 100, 1)


# ===========================================
# API Endpoints
# ===========================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "decomposer": DECOMPOSER_URL,
        "decomposer_cot": DECOMPOSER_COT_URL,
        "verifier": VERIFIER_URL,
        "verifier_cot": VERIFIER_COT_URL
    }


@app.post("/check", response_model=CheckResponse)
async def check_facts(req: CheckRequest):
    """Main endpoint: decompose → retrieve → verify → score."""
    
    use_reasoning = req.use_reasoning
    mode = "reasoning" if use_reasoning else "standard"
    
    # Step 1: Decompose text into claims
    raw_claims = await decompose_text(req.text, use_reasoning)
    
    if not raw_claims:
        return CheckResponse(
            claims=[],
            factuality_score=100.0,
            supported_count=0,
            refuted_count=0,
            nei_count=0,
            opinion_count=0,
            mode=mode
        )
    
    # Step 2-3: For each claim, retrieve evidence and verify
    results: List[ClaimResult] = []
    
    for claim_data in raw_claims:
        claim_result = ClaimResult(
            quote=claim_data.get("quote", ""),
            atomic_claim=claim_data.get("atomic_claim", ""),
            claim_type=claim_data.get("type", "FACTUAL"),
            reason=claim_data.get("reason") if use_reasoning else None
        )
        
        # Skip verification for opinions
        if claim_result.claim_type == "OPINION":
            claim_result.verdict = "OPINION"
            claim_result.confidence = 1.0
            claim_result.evidence = "Subjective opinion - not fact-checked"
            results.append(claim_result)
            continue
        
        # Retrieve evidence
        evidence, source = await retrieve_evidence(claim_result.atomic_claim)
        claim_result.evidence = evidence
        claim_result.source = source
        
        # Verify claim
        if evidence and not evidence.startswith("No Tavily") and not evidence.startswith("Evidence retrieval"):
            verdict, confidence, reasoning = await verify_claim(
                claim_result.atomic_claim, 
                evidence, 
                use_reasoning
            )
            claim_result.verdict = verdict
            claim_result.confidence = confidence
            if reasoning:
                claim_result.reasoning = reasoning
        else:
            claim_result.verdict = "NOT_ENOUGH_INFO"
            claim_result.confidence = 0.5
        
        results.append(claim_result)
    
    # Step 4: Calculate factuality score
    factuality_score = calculate_factuality_score(results)
    
    # Count verdicts
    supported = sum(1 for c in results if c.verdict == "SUPPORTED")
    refuted = sum(1 for c in results if c.verdict == "REFUTED")
    nei = sum(1 for c in results if c.verdict == "NOT_ENOUGH_INFO")
    opinions = sum(1 for c in results if c.verdict == "OPINION")
    
    return CheckResponse(
        claims=results,
        factuality_score=factuality_score,
        supported_count=supported,
        refuted_count=refuted,
        nei_count=nei,
        opinion_count=opinions,
        mode=mode
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
