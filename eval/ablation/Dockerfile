FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir httpx

COPY grammars/ /app/grammars/
COPY run_ablation.py /app/

CMD ["python", "run_ablation.py", \
     "--task", "both", \
     "--mode", "cot", \
     "--decomp-url", "http://decomposer-cot:8080", \
     "--verif-url", "http://verifier-cot:8080", \
     "--output-dir", "/app/outputs/ablation"]
