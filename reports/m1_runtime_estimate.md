# M1/Ollama runtime estimates for the P1 expanded evaluation

Measured 2026-10-28 on Apple M1 (8 cores, 16 GiB unified), macOS 27.2, Ollama
0.40.12, 100 stratified manifest prompts per model (`src/benchmark/ollama_speed_test.py`;
raw rows in `m1_ollama_benchmark.csv` + `m1_ollama_benchmark_14b.csv`). All models
Q4_K_M GGUF; options: temperature 0, top_k 1, seed 42, num_ctx 4096, num_predict 32.
Planned manifest: 16,618 calls per model. Estimates: optimistic = median call time,
median = measured sample wall-clock per call, conservative = p90 call time.

| Model | Digest | Median call | Mean call | Output tok/s | Parse-valid | Context trunc. | Peak swap | Optimistic | Median | Conservative |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | a8b0c515… | 0.33 s | 0.69 s | ~235 | 100/100 | 0 | 3.0 GiB | 1.5 h | **3.3 h** | 8.1 h |
| qwen2.5:3b | 357c53fb… | 2.0 s | 4.1 s | ~41 | 100/100 | 0 | ~3 GiB | 9.7 h | **19.5 h** | 52.0 h |
| qwen2.5:7b | c0ed8a1d… (see CSV) | 5.0 s | 9.8 s | 16.8 | 99/100 | 0 | 7.5 GiB | 23.1 h | **45.4 h** | 119.7 h |
| qwen2.5:14b | 7cdf5a01… | 16.4 s | 27.3 s | 5.9 | 99/100 | 0 | **17.9 GiB** | 75.8 h | **126.7 h (~5.3 days)** | 338.5 h (~14 days) |

Memory reality on 16 GiB: the 7b run already drove swap to 7.5 GiB with free
memory bottoming near 50 MB; the 14b run swapped **more than physical RAM**
(17.9 GiB) with free memory at 16 MB, and its throughput (5.9 tok/s generation,
44.6 tok/s prefill) is bandwidth/swap-bound. The machine remains usable but
degraded during 7b, and severely degraded during 14b.

Disk: results are JSONL with raw responses, ≈ 25–40 MB per model; model weights
already reside on the external store. No additional meaningful disk pressure.

## Recommended evaluation matrix

| Model | Planned calls | Est. runtime (median) | Memory feasibility | Recommendation |
|---|---|---|---|---|
| qwen2.5:0.5b | 16,618 | 3.3 h | verified comfortable | **Include** |
| qwen2.5:3b | 16,618 | 19.5 h | verified comfortable | **Include** |
| qwen2.5:7b | 16,618 | 45.4 h | verified marginal (7.5 GiB swap) | **Include** (run unattended, machine otherwise idle) |
| qwen2.5:14b | 16,618 | 126.7 h | measured: exceeds RAM, 17.9 GiB swap | **User-directed include (full manifest)**; measured alternative: paired subset (e.g. nine_way component only, 12,150 calls ≈ 92 h, or one 25%-of-groups paired subsample ≈ 32 h) |

Sequential execution order: 0.5b → 3b → 7b → 14b, with per-model checkpointed
JSONL (resume-safe, duplicate-proof). Total for all four at median estimates:
**≈ 195 hours (~8.1 days) of continuous runtime**, dominated by the 14b leg.

## Optional GPT-4o arm (requires separate explicit approval; never auto-launched)

Token basis: measured prompt-token counts on the benchmark sample, scaled to the
manifest by calibrated characters-per-token (Qwen tokenizer as proxy; o200k differs
by ~±20%): ≈ 20.1 M input tokens + ≈ 0.13 M output tokens. At $2.50/M input and
$10/M output (pricing as of 2026-10): **≈ $51 (range ≈ $40–65)** for all 16,618
calls. The audited registered-provider pipeline (plan → review gate → execute)
remains the sanctioned route for any GPT-4o run above 2,000 calls.
