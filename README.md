# Qwen3 Embedding 0.6B on DGX Spark

An OpenAI-compatible embedding service for the NVIDIA GB10 GPU. vLLM loads the
official [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
Hugging Face checkpoint, converts its linear weights to FP8 at load time, and
serves it under the exact same model ID. Each embedding has 1,024 dimensions.

## Start and stop

Prerequisites: DGX Spark or a compatible GB10 host, Docker with the NVIDIA
container runtime, and Docker Compose with GPU support.

```sh
cp .env.example .env
chmod 600 .env
# Edit .env and set LLM_API_KEY before starting.
./start.sh
docker compose ps
docker compose logs -f embedding
./stop.sh
```

The default listener is `127.0.0.1:8891`. To reach it from another machine,
set `BIND_HOST` to a private interface address. Set `PORT` if 8891 is taken.
`start.sh` checks whether the chosen port is free. Keep `.env` local; Git
ignores it. Mounts preserve the Hugging Face and vLLM caches across restarts.

The default image is `nvcr.io/nvidia/vllm:26.09-py3`. It was tested with GB10
driver 595.99.02 using Triton attention and FlashInfer autotuning disabled.
The image's default attention path failed during startup with a PTX toolchain
error on that driver.

## Call the API

Set `LLM_API_KEY` in your client environment to the key configured on the server.
Use `http://127.0.0.1:8891/v1` as the OpenAI-compatible base URL when running
locally.

```sh
BASE_URL=http://127.0.0.1:8891

curl -fsS "$BASE_URL/v1/models" \
  -H "Authorization: Bearer $LLM_API_KEY"

curl -fsS "$BASE_URL/v1/embeddings" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":["Clausola di rinnovo","Renewal clause"]}'
```

Omit the optional `dimensions` field. Keep the same embedding configuration for
indexing and querying; rebuild an existing index if switching from BF16 to FP8.

## Memory and batching

GB10 shares its memory with the CPU. These numbers describe different layers
and should not be added to the container reading.

| Memory item | Observed or configured |
| --- | ---: |
| Loaded FP8 model weights | 0.74 GiB, down from 1.12 GiB with BF16 |
| Explicit KV cache reservation | 1.25 GiB, down from 3.95 GiB with automatic sizing |
| Container-accounted memory immediately after benchmark | 3.17 GiB |
| Container-accounted peak since start, including benchmark | 3.38 GiB |
| Docker memory limit | 12 GiB |

The two measured model allocations fell by about **3.1 GiB** combined. The
container figures come from cgroup memory accounting and are not a GPU VRAM
reading or a hard GPU cap. `KV_CACHE_MEMORY_BYTES` sets the fixed cache
reservation. This version still uses `GPU_MEMORY_UTILIZATION` (5% by default)
for startup admission, even with an explicit cache. The default scheduler
allows up to 8,192 tokens per input and per scheduling batch and eight sequences.
The cache holds 11,696 resident tokens in this configuration, so concurrent
long inputs can queue.

## Benchmarks

One DGX Spark GB10, FP8 weights, 1.25 GiB KV cache, eager execution. A same-host
HTTP client sent synthetic text to the private listener. Each workload used
one warm-up and 20 sequential timed requests with distinct inputs. Latency
includes the HTTP round trip and response decoding. Throughput is aggregate input
tokens divided by the timed wall time for that workload.

| Workload | Median input tokens per request | p50 | p95 | Input tokens/s |
| --- | ---: | ---: | ---: | ---: |
| One short query | 21 | 13.0 ms | 15.9 ms | 1,513 |
| Eight passages in one request | 7,280 | 176.0 ms | 180.6 ms | 41,297 |
| One near-limit input | 7,947 | 413.8 ms | 421.3 ms | 19,157 |

```sh
python3 benchmark.py --base-url "$BASE_URL" --requests 20
```

The benchmark checks that requests without a key receive HTTP 401, verifies
the served model ID, and validates each returned vector. These figures measure
the embedding service; archive indexing includes additional work for parsing,
chunking, network transfer, and index writes. The benchmark does not score
retrieval quality.
