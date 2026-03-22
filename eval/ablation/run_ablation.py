"""
Vera CFG Ablation Study
========================
Measures the impact of GBNF grammar-constrained decoding on:
  1. JSON validity rate
  2. Schema correctness (required fields present)
  3. Quote alignment accuracy (quotes are exact substrings of input)
  4. Verification accuracy (for verifier ablation)

Runs each task in two conditions:
  - WITH grammar:    constrained decoding via llama.cpp `grammar` parameter
  - WITHOUT grammar: unconstrained generation (model alone)

Usage:
  # Against running services (see docker-compose.ablation.yml):
  python run_ablation.py --task decomp --sample-size 100

  # Or point at custom URLs:
  python run_ablation.py --task decomp --url http://localhost:8080 \
                         --sample-size 100 --data ../data/distilled_cot/vera_test.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED = 42
SAMPLE_SIZE = 100
OUTPUT_DIR = Path("./outputs/ablation")

GRAMMAR_DIR = Path(__file__).parent / "grammars"

# ---------------------------------------------------------------------------
# System prompts (same as production)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def load_grammar(name: str) -> str:
    """Load a GBNF grammar file by name."""
    path = GRAMMAR_DIR / f"{name}.gbnf"
    if not path.exists():
        print(f"ERROR: Grammar file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def wait_for_service(url: str, name: str, timeout: int = 300):
    print(f"Waiting for {name} at {url} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  {name} is ready.", flush=True)
                return
        except httpx.ConnectError:
            pass
        time.sleep(3)
    print(f"  WARNING: {name} not healthy within {timeout}s, proceeding.", flush=True)


def write_jsonl(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} records -> {path}", flush=True)


# ---------------------------------------------------------------------------
# Quote alignment (same cascade as production)
# ---------------------------------------------------------------------------

def check_quote_alignment(quote: str, source_text: str, threshold: float = 0.85) -> bool:
    """Check if quote aligns to source text using the 4-strategy cascade."""
    if not quote or not source_text:
        return False

    # Strategy 1: Exact match
    if quote in source_text:
        return True

    # Strategy 2: Case-insensitive
    if quote.lower() in source_text.lower():
        return True

    # Strategy 3: Fuzzy sliding-window match
    q_len = len(quote)
    if q_len > 0:
        for start in range(len(source_text) - q_len + 1):
            window = source_text[start : start + q_len]
            ratio = SequenceMatcher(None, quote, window).ratio()
            if ratio >= threshold:
                return True

    # Strategy 4: Ellipsis handling
    if "..." in quote:
        segments = [s.strip() for s in quote.split("...") if s.strip()]
        if segments and all(seg in source_text for seg in segments):
            return True

    return False


# ---------------------------------------------------------------------------
# Call the LLM (with or without grammar)
# ---------------------------------------------------------------------------

def call_llm(
    client: httpx.Client,
    url: str,
    system_prompt: str,
    user_content: str,
    grammar: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request, optionally with GBNF grammar."""
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if grammar:
        payload["grammar"] = grammar

    resp = client.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Decomposition ablation
# ---------------------------------------------------------------------------

DECOMP_REQUIRED_FIELDS_COT = {"quote", "atomic_claim", "reason", "type"}
DECOMP_REQUIRED_FIELDS_STD = {"quote", "atomic_claim", "type"}
VALID_TYPES = {"FACTUAL", "OPINION"}


def evaluate_decomp_output(raw: str, input_text: str, is_cot: bool) -> dict:
    """Evaluate a single decomposition output for validity metrics."""
    result = {
        "json_valid": False,
        "schema_valid": False,
        "is_array": False,
        "num_claims": 0,
        "quotes_total": 0,
        "quotes_aligned": 0,
        "type_valid": True,
        "parse_error": None,
    }

    required = DECOMP_REQUIRED_FIELDS_COT if is_cot else DECOMP_REQUIRED_FIELDS_STD

    # Try to parse JSON
    try:
        # Try direct parse first
        parsed = json.loads(raw)
        result["json_valid"] = True
    except json.JSONDecodeError:
        # Try extracting JSON array from surrounding text
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                result["json_valid"] = True
            else:
                result["parse_error"] = "No JSON array found"
                return result
        except json.JSONDecodeError as e:
            result["parse_error"] = str(e)
            return result

    # Check array
    if not isinstance(parsed, list):
        result["parse_error"] = f"Expected array, got {type(parsed).__name__}"
        return result
    result["is_array"] = True
    result["num_claims"] = len(parsed)

    # Check each claim
    schema_ok = True
    for claim in parsed:
        if not isinstance(claim, dict):
            schema_ok = False
            continue
        if not required.issubset(claim.keys()):
            schema_ok = False
        if claim.get("type", "").upper() not in VALID_TYPES:
            result["type_valid"] = False

        # Quote alignment
        quote = claim.get("quote", "")
        if quote:
            result["quotes_total"] += 1
            if check_quote_alignment(quote, input_text):
                result["quotes_aligned"] += 1

    result["schema_valid"] = schema_ok and len(parsed) > 0
    return result


def run_decomp_ablation(
    client: httpx.Client,
    url: str,
    samples: list[dict],
    grammar: str | None,
    condition: str,
    is_cot: bool,
) -> list[dict]:
    """Run decomposition on all samples and evaluate."""
    results = []
    total = len(samples)
    prompt = DECOMPOSER_PROMPT_COT if is_cot else DECOMPOSER_PROMPT_STANDARD

    for i, sample in enumerate(samples, 1):
        input_text = sample["input"]
        print(f"  [{condition}] Decomposing [{i}/{total}]: {input_text[:60]}...", flush=True)

        try:
            raw = call_llm(client, url, prompt, input_text, grammar=grammar)
            metrics = evaluate_decomp_output(raw, input_text, is_cot)
            results.append({
                "id": i,
                "condition": condition,
                "input": input_text[:200],
                "raw_output": raw,
                **metrics,
            })
        except Exception as e:
            results.append({
                "id": i,
                "condition": condition,
                "input": input_text[:200],
                "raw_output": "",
                "json_valid": False,
                "schema_valid": False,
                "is_array": False,
                "num_claims": 0,
                "quotes_total": 0,
                "quotes_aligned": 0,
                "type_valid": False,
                "parse_error": f"Request error: {e}",
            })

    return results


# ---------------------------------------------------------------------------
# Verification ablation
# ---------------------------------------------------------------------------

def evaluate_verif_output(raw: str) -> dict:
    """Evaluate a single verification output for validity metrics."""
    result = {
        "json_valid": False,
        "schema_valid": False,
        "has_reasoning": False,
        "has_verdict": False,
        "verdict_valid": False,
        "predicted_verdict": None,
        "parse_error": None,
    }

    try:
        parsed = json.loads(raw)
        result["json_valid"] = True
    except json.JSONDecodeError:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                result["json_valid"] = True
            else:
                result["parse_error"] = "No JSON object found"
                return result
        except json.JSONDecodeError as e:
            result["parse_error"] = str(e)
            return result

    if not isinstance(parsed, dict):
        result["parse_error"] = f"Expected object, got {type(parsed).__name__}"
        return result

    result["has_reasoning"] = "reasoning" in parsed and bool(parsed["reasoning"])
    result["has_verdict"] = "verdict" in parsed
    result["schema_valid"] = result["has_reasoning"] and result["has_verdict"]

    verdict = parsed.get("verdict", "").upper()
    result["predicted_verdict"] = verdict
    result["verdict_valid"] = verdict in {"SUPPORTED", "REFUTED"}

    return result


def run_verif_ablation(
    client: httpx.Client,
    url: str,
    samples: list[dict],
    grammar: str | None,
    condition: str,
) -> list[dict]:
    """Run verification on all samples and evaluate."""
    results = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        inp = sample["input"]
        claim = inp["claim"]
        evidence_list = inp.get("evidence", [])
        evidence_str = "\n".join(f"Evidence {j+1}: {e}" for j, e in enumerate(evidence_list))
        gt_label = sample.get("ground_truth", "")
        user_content = f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_str}"

        print(f"  [{condition}] Verifying [{i}/{total}]: {claim[:60]}...", flush=True)

        try:
            raw = call_llm(client, url, VERIFIER_PROMPT_COT, user_content,
                           grammar=grammar, max_tokens=512)
            metrics = evaluate_verif_output(raw)
            results.append({
                "id": i,
                "condition": condition,
                "claim": claim[:200],
                "ground_truth": gt_label,
                "raw_output": raw,
                **metrics,
            })
        except Exception as e:
            results.append({
                "id": i,
                "condition": condition,
                "claim": claim[:200],
                "ground_truth": gt_label,
                "raw_output": "",
                "json_valid": False,
                "schema_valid": False,
                "has_reasoning": False,
                "has_verdict": False,
                "verdict_valid": False,
                "predicted_verdict": None,
                "parse_error": f"Request error: {e}",
            })

    return results


# ---------------------------------------------------------------------------
# Aggregate & print summary
# ---------------------------------------------------------------------------

def summarise_decomp(results: list[dict], condition: str):
    n = len(results)
    if n == 0:
        return
    json_ok = sum(r["json_valid"] for r in results)
    schema_ok = sum(r["schema_valid"] for r in results)
    qt = sum(r["quotes_total"] for r in results)
    qa = sum(r["quotes_aligned"] for r in results)
    type_ok = sum(r["type_valid"] for r in results)

    print(f"\n{'=' * 55}")
    print(f"  DECOMPOSITION ABLATION — {condition.upper()}")
    print(f"{'=' * 55}")
    print(f"  Samples:             {n}")
    print(f"  JSON valid:          {json_ok}/{n}  ({json_ok/n*100:.1f}%)")
    print(f"  Schema correct:      {schema_ok}/{n}  ({schema_ok/n*100:.1f}%)")
    print(f"  Type field valid:    {type_ok}/{n}  ({type_ok/n*100:.1f}%)")
    if qt > 0:
        print(f"  Quote alignment:     {qa}/{qt}  ({qa/qt*100:.1f}%)")
    else:
        print(f"  Quote alignment:     N/A (no quotes extracted)")
    print(f"{'=' * 55}")


def summarise_verif(results: list[dict], condition: str):
    n = len(results)
    if n == 0:
        return
    json_ok = sum(r["json_valid"] for r in results)
    schema_ok = sum(r["schema_valid"] for r in results)
    verdict_ok = sum(r["verdict_valid"] for r in results)
    has_gt = [r for r in results if r.get("ground_truth")]
    correct = sum(
        1 for r in has_gt
        if r["predicted_verdict"] and r["predicted_verdict"] == r["ground_truth"].upper()
    )

    print(f"\n{'=' * 55}")
    print(f"  VERIFICATION ABLATION — {condition.upper()}")
    print(f"{'=' * 55}")
    print(f"  Samples:             {n}")
    print(f"  JSON valid:          {json_ok}/{n}  ({json_ok/n*100:.1f}%)")
    print(f"  Schema correct:      {schema_ok}/{n}  ({schema_ok/n*100:.1f}%)")
    print(f"  Verdict valid:       {verdict_ok}/{n}  ({verdict_ok/n*100:.1f}%)")
    if has_gt:
        print(f"  Accuracy (vs GT):    {correct}/{len(has_gt)}  ({correct/len(has_gt)*100:.1f}%)")
    print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vera CFG Ablation Study")
    parser.add_argument("--task", choices=["decomp", "verif", "both"], default="both",
                        help="Which task to ablate")
    parser.add_argument("--mode", choices=["cot", "standard"], default="cot",
                        help="Decomposer mode (cot or standard)")
    parser.add_argument("--url", default=None,
                        help="LLM server URL (overrides per-task defaults)")
    parser.add_argument("--decomp-url", default=os.getenv("DECOMPOSER_URL", "http://localhost:8080"),
                        help="Decomposer llama.cpp URL")
    parser.add_argument("--verif-url", default=os.getenv("VERIFIER_URL", "http://localhost:8080"),
                        help="Verifier llama.cpp URL")
    parser.add_argument("--decomp-data",
                        default=os.getenv("DECOMP_DATA", "../data/distilled_cot/vera_test.jsonl"))
    parser.add_argument("--verif-data",
                        default=os.getenv("VERIF_DATA", "../data/distilled_verification/vera_test.jsonl"))
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = httpx.Client()

    print("=" * 60, flush=True)
    print("Vera CFG Ablation Study", flush=True)
    print("=" * 60, flush=True)
    print(f"  Tasks:       {args.task}", flush=True)
    print(f"  Mode:        {args.mode}", flush=True)
    print(f"  Samples:     {args.sample_size}", flush=True)
    print(f"  Seed:        {args.seed}", flush=True)
    print(f"  Output:      {output_dir}", flush=True)
    print(flush=True)

    # ---- Decomposition ablation ----
    if args.task in ("decomp", "both"):
        decomp_path = Path(args.decomp_data)
        decomp_url = args.url or args.decomp_url
        is_cot = args.mode == "cot"
        grammar_name = f"decomposer_{args.mode}"

        if not decomp_path.exists():
            print(f"ERROR: Data not found: {decomp_path}", file=sys.stderr)
        else:
            wait_for_service(decomp_url, "Decomposer")
            data = load_jsonl(decomp_path)
            samples = sample_items(data, args.sample_size, args.seed)

            grammar_text = load_grammar(grammar_name)

            print(f"\n--- Decomposition WITH grammar ({len(samples)} samples) ---", flush=True)
            res_with = run_decomp_ablation(client, decomp_url, samples, grammar_text,
                                           "with_grammar", is_cot)

            print(f"\n--- Decomposition WITHOUT grammar ({len(samples)} samples) ---", flush=True)
            res_without = run_decomp_ablation(client, decomp_url, samples, None,
                                              "without_grammar", is_cot)

            write_jsonl(output_dir / f"decomp_{args.mode}_with_grammar.jsonl", res_with)
            write_jsonl(output_dir / f"decomp_{args.mode}_without_grammar.jsonl", res_without)

            summarise_decomp(res_with, "WITH grammar")
            summarise_decomp(res_without, "WITHOUT grammar")

    # ---- Verification ablation ----
    if args.task in ("verif", "both"):
        verif_path = Path(args.verif_data)
        verif_url = args.url or args.verif_url

        if not verif_path.exists():
            print(f"ERROR: Data not found: {verif_path}", file=sys.stderr)
        else:
            wait_for_service(verif_url, "Verifier")
            data = load_jsonl(verif_path)
            samples = sample_items(data, args.sample_size, args.seed)

            grammar_text = load_grammar("verifier_cot")

            print(f"\n--- Verification WITH grammar ({len(samples)} samples) ---", flush=True)
            res_with = run_verif_ablation(client, verif_url, samples, grammar_text,
                                          "with_grammar")

            print(f"\n--- Verification WITHOUT grammar ({len(samples)} samples) ---", flush=True)
            res_without = run_verif_ablation(client, verif_url, samples, None,
                                              "without_grammar")

            write_jsonl(output_dir / f"verif_cot_with_grammar.jsonl", res_with)
            write_jsonl(output_dir / f"verif_cot_without_grammar.jsonl", res_without)

            summarise_verif(res_with, "WITH grammar")
            summarise_verif(res_without, "WITHOUT grammar")

    # ---- Combined summary table ----
    print("\n\n" + "=" * 60, flush=True)
    print("CFG ABLATION COMPLETE", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
