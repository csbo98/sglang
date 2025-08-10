### YAML configuration for sglang.launch_server

- Run with a single YAML file:

```bash
python3 -m sglang.launch_server --config config.yaml
```

- CLI overrides YAML. You can still pass flags alongside `--config` and they win:

```bash
python3 -m sglang.launch_server --config config.yaml --port 31000 --log-level debug
```

- You can also set `SGLANG_CONFIG=/path/to/config.yaml` to use a config by default.

- Supported keys: All existing server flags. Use either the long flag name with dashes (e.g., `model-path`) or the Python-style name with underscores (e.g., `model_path`). Short aliases are supported for parallel sizes: `tp`, `pp`, `dp`, `ep`.

- Example file: see `docs/server_config_example.yaml`.

### YAML for router+server launcher

For the combined launcher `sgl-router/py_src/sglang_router/launch_server.py`, provide a single YAML with two sections:

```yaml
server:
  model: meta-llama/Meta-Llama-3-8B-Instruct
  tp: 2
  host: 0.0.0.0
  port: 30000

router:
  worker-urls:
    - http://127.0.0.1:31001
    - http://127.0.0.1:31002
  policy: cache_aware
```

CLI flags still override YAML values.