### Multi‑Turn Chat Benchmark — Overview

This directory contains a lightweight benchmark to stress-test multi‑turn chat style generation across different backends (sglang, vLLM, guidance). It simulates realistic conversations with multiple turns per request, configurable prompt/answer lengths, and reports end‑to‑end latency for a batch of concurrent sessions.

### What it measures

- **End‑to‑end latency**: Wall‑clock time to complete N chat sessions, each with T turns.
- **Backend throughput under conversational load**: Repeated prompt→generate cycles per session.
- **Effect of prompt/answer length**: Short vs long outputs; optional very long system prompt variant.

### File map

- **`bench_sglang.py`**: Runs the benchmark using sglang’s program API (`@sgl.function`) and `run_batch`.
- **`bench_other.py`**: Runs the same workload on non‑sglang backends (e.g., vLLM, guidance) via a unified `generate` call.
- **`data_gen.py`**: Deterministic synthetic data generator for prompts/lengths (token‑aware).
- **`long_prompt_multi_turn.py`**: sglang benchmark variant that prepends a long system prompt and caches generated inputs.
- **`README.md`**: Quick commands to launch servers and run the benchmarks.

### Core concepts and design

- **Synthetic, token‑aware prompts**
  - `data_gen.py` produces pseudo‑random alphanumeric prompts targeted to a token budget using the provided tokenizer.
  - Each request is a session with `turns` Q→A pairs; for each turn we append a new prompt and request `new_tokens` for the answer.

- **Two execution paths (same workload)**
  - **sglang (`bench_sglang.py`)**: Defines a program with `@sgl.function` that loops over turns and calls `sgl.gen`. Runs all sessions via `run_batch`, with `num_threads` controlling host parallelism.
  - **Other backends (`bench_other.py`)**: Wraps a backend‑specific `generate` function (e.g., REST/gRPC/local call). It builds the conversation string incrementally and executes per‑request in a thread pool (`--parallel`).

- **Long system prompt variant**
  - `long_prompt_multi_turn.py` prepends a large system prompt to each session and fixes Q/A lengths (`--system-prompt-len`, `--len-q`, `--len-a`).
  - Inputs are cached under `~/.cache/sglang` keyed by parameters and tokenizer to avoid regeneration and to improve reproducibility.

- **Metrics & outputs**
  - Latency is measured with `time.perf_counter()` from the start of the batch until all sessions complete.
  - Raw per‑session outputs are dumped to `tmp_output_{backend}.txt`.
  - A JSON line with the run summary (task, backend, parallelism, turns, counts, latency) is appended to `--result-file` (provided by the common arg parsers).

### CLI parameters (high‑level)

- **Workload shape**
  - `--turns`: Number of Q→A turns per session.
  - `--num-qa`: Number of sessions (requests).
  - Short/long answer modes:
    - Default short answers: small `new_tokens` range.
    - `--long`: increases answer length ranges (see each script’s defaults below).

- **Lengths (token counts)**
  - `bench_sglang.py` / `bench_other.py`:
    - `--min-len-q`, `--max-len-q`: Prompt token range per turn.
    - `--min-len-a`, `--max-len-a`: Answer token range per turn.
  - `long_prompt_multi_turn.py`:
    - `--system-prompt-len`: Tokens in the system prompt prefix.
    - `--len-q`: Tokens per user prompt (fixed per turn).
    - `--len-a`: Tokens to generate per turn.

- **Backend and tokenizer**
  - `--backend`: Selected by the common parsers (sglang local/server, vLLM, guidance, etc.).
  - `--tokenizer`: Hugging Face tokenizer ID (e.g., `meta-llama/Llama-2-7b-chat-hf`).
  - `--trust-remote-code`: Forwarded to tokenizer loaders if needed.

- **Parallelism**
  - sglang: `run_batch(..., num_threads=...)` (auto or explicit `--parallel`).
  - other backends: `--parallel` threads via `ThreadPoolExecutor`.

### Defaults (quick reference)

- `bench_sglang.py` and `bench_other.py`:
  - Defaults: `--turns 4`, `--num-qa 20`, prompts 256–512 tokens, answers 4–8 tokens.
  - With `--long`: answers 256–512 tokens, `--num-qa 20`.

- `long_prompt_multi_turn.py`:
  - Defaults: `--turns 8`, `--num-qa 128`, `--system-prompt-len 2048`, `--len-q 32`, `--len-a 128`.

### Quickstart

- **sglang backend**
  1) Launch a model server (example: Llama‑2‑7B chat):
```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```
  2) Run the benchmark (short answers):
```bash
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf
```
  3) Long answers:
```bash
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf --long
```

- **vLLM backend**
  1) Launch vLLM API server:
```bash
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```
  2) Run the benchmark:
```bash
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm
```
  3) Long answers:
```bash
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm --long
```

- **guidance backend** (example flags; adjust `--model-path` and context size):
```bash
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

- **Long system prompt variant (sglang)**
```bash
python3 long_prompt_multi_turn.py --tokenizer meta-llama/Meta-Llama-3-8B-Instruct
```

### Notes and tips

- **Determinism**: `data_gen.py` seeds RNG (`random.seed(42)`) for reproducible workloads. The long system prompt variant caches generated inputs to `~/.cache/sglang` keyed by parameters and tokenizer.
- **EOS handling**: sglang calls use `ignore_eos=True` to force generation up to the requested token count.
- **Temperature**: All scripts set `temperature=0` for stable comparisons.
- **Results**: Check `tmp_output_{backend}.txt` for raw generations and `--result-file` (JSONL) for summary rows.
- **Scaling**: Increase `--num-qa` and `--turns` to stress throughput; tune `--parallel` / `num_threads` to match CPU cores.
- **Memory**: For large MoE models (e.g., Mixtral‑8x7B), consult the example flags in `README.md` and consider reducing static memory fractions if needed.

