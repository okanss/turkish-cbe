# Hardware recommendations — P1 local evaluation on Apple M1 (16 GiB)

Profile: `reports/hardware_profile.json` (`src/hardware_profile.py`). Measured
throughput: `reports/m1_runtime_estimate.md`.

Detected: Apple M1, 8 cores (4P+4E), 16 GiB unified memory, macOS 27.2, Ollama
0.40.12 (server running), model store on the external volume via
`~/.ollama/models` symlink. Reclaimable memory at profiling time ≈ 5.9 GiB; the
static feasibility rule (weights ≤ 60% of RAM and weights + 3 GiB headroom within
reclaimable memory) classified 0.5b and 3b as feasible, 7b feasible after pull
(4.4 GiB weights), and 14b as marginal pending measurement — confirmed by the
benchmark: 7b swaps up to 7.5 GiB; 14b swaps beyond physical RAM (17.9 GiB) and
generates at 5.9 tok/s.

Recommendations:

1. **Primary evaluation on qwen2.5:0.5b, 3b, 7b** (exact digests and Q4_K_M
   quantization recorded per run). Run sequentially, machine otherwise idle;
   keep the external model store connected for the entire run.
2. **14b at full manifest only as a deliberate, user-directed choice**: measured
   ≈ 5.3 days median (up to ~14 days at p90) with the machine severely degraded;
   checkpointing makes interruption safe, so it can run in night-time slices.
   The measured cheaper alternatives are in the runtime report.
3. Keep `num_ctx = 4096`: zero context truncations observed across 400 measured
   calls up to n = 128 (max observed prompt 4,023 tokens).
4. Disable system sleep for unattended runs (`caffeinate -is <command>` wraps the
   runner); the runner resumes exactly where it stopped if anything interrupts it.
5. Quantization caveat (manuscript-bound): Q4_K_M results are a supplemental
   quantized arm; they are never comparable one-to-one with full-precision
   registered-provider results and are labeled as such in every artifact.
