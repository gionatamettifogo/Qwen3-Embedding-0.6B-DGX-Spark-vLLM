#!/usr/bin/env python3
"""Time authenticated embedding requests with no external Python packages."""

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIMENSION = 1024


def request_json(base, path, key=None, body=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def verify_authentication(base, key):
    for path, body in [
        ("/v1/models", None),
        ("/v1/embeddings", {"model": MODEL, "input": "Authentication check"}),
    ]:
        try:
            request_json(base, path, body=body)
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise
        else:
            raise RuntimeError(f"{path} allowed a request without an API key")
    models = request_json(base, "/v1/models", key=key)
    if [model["id"] for model in models["data"]] != [MODEL]:
        raise RuntimeError("The served model name does not match the Hugging Face base model")


def embed(base, key, inputs):
    result = request_json(base, "/v1/embeddings", key=key, body={"model": MODEL, "input": inputs})
    returned = result["data"]
    if result["model"] != MODEL or [row["index"] for row in returned] != list(range(len(inputs))):
        raise RuntimeError("Unexpected model name or embedding count")
    if any(len(row["embedding"]) != DIMENSION or not all(map(math.isfinite, row["embedding"]))
           for row in returned):
        raise RuntimeError("Invalid embedding vector")
    return result["usage"]["prompt_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8891", help="Server URL without /v1")
    parser.add_argument("--requests", type=int, default=20, help="Timed requests per workload")
    args = parser.parse_args()
    if not 1 <= args.requests <= 100:
        parser.error("--requests must be between 1 and 100")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("VLLM_API_KEY")
    if not key:
        parser.error("Set LLM_API_KEY before running the benchmark")
    base = args.base_url.rstrip("/")
    verify_authentication(base, key)
    profiles = {
        "short query": lambda i: [f"Richiesta {i}: trova la clausola di rinnovo del contratto."],
        "8 passages": lambda i: [f"Passaggio {i}-{j}: " + " memoria e clausole" * 180 for j in range(8)],
        "near-limit input": lambda i: [f"Test {i}: " + " prova" * 7940],
    }
    summary = []
    for label, make_inputs in profiles.items():
        embed(base, key, make_inputs(0))
        times, token_counts = [], []
        for i in range(1, args.requests + 1):
            started = time.perf_counter()
            tokens = embed(base, key, make_inputs(i))
            times.append(time.perf_counter() - started)
            token_counts.append(tokens)
        ordered = sorted(times)
        summary.append({
            "workload": label,
            "requests": args.requests,
            "median_input_tokens_per_request": statistics.median(token_counts),
            "median_ms": round(1000 * statistics.median(times), 1),
            "p95_ms": round(1000 * ordered[math.ceil(0.95 * len(ordered)) - 1], 1),
            "input_tokens_per_second": round(sum(token_counts) / sum(times)),
        })
    print(json.dumps({"model": MODEL, "dimensions": DIMENSION, "authentication_checked": True,
                      "results": summary}, indent=2))


if __name__ == "__main__":
    main()
