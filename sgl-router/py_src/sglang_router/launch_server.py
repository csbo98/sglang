import argparse
import copy
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from typing import List

import requests
from setproctitle import setproctitle
from sglang_router.launch_router import RouterArgs, launch_router

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available
from typing import Any, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


# Create new process group
def run_server(server_args, dp_rank):
    """
    Note:

    1. Without os.setpgrp(), all processes share the same PGID. When you press Ctrl+C, the terminal sends SIGINT to all processes in the group simultaneously.
    This can cause leaf processes to terminate first, which messes up the cleaning order and produces orphaned processes.

    Terminal (PGID=100)
    └── Main Python Process (PGID=100)
        └── Server Process 1 (PGID=100)
            └── Scheduler 1
            └── Detokenizer 1
        └── Server Process 2 (PGID=100)
            └── Scheduler 2
            └── Detokenizer 2

    2. With os.setpgrp(), the main Python process and its children are in a separate group. Now:

    Terminal (PGID=100)
    └── Main Python Process (PGID=200)
        └── Server Process 1 (PGID=300)
            └── Scheduler 1
            └── Detokenizer 1
        └── Server Process 2 (PGID=400)
            └── Scheduler 2
            └── Detokenizer 2
    """
    # create new process group
    os.setpgrp()

    setproctitle("sglang::server")
    # Set SGLANG_DP_RANK environment variable
    os.environ["SGLANG_DP_RANK"] = str(dp_rank)

    launch_server(server_args)


def launch_server_process(
    server_args: ServerArgs, worker_port: int, dp_id: int
) -> mp.Process:
    """Launch a single server process with the given args and port."""
    server_args = copy.deepcopy(server_args)
    server_args.port = worker_port
    server_args.base_gpu_id = dp_id * server_args.tp_size
    server_args.dp_size = 1

    proc = mp.Process(target=run_server, args=(server_args, dp_id))
    proc.start()
    return proc


def wait_for_server_health(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for server to be healthy by checking /health endpoint."""
    start_time = time.perf_counter()
    url = f"http://{host}:{port}/health"

    while time.perf_counter() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def find_available_ports(base_port: int, count: int) -> List[int]:
    """Find consecutive available ports starting from base_port."""
    available_ports = []
    current_port = base_port

    while len(available_ports) < count:
        if is_port_available(current_port):
            available_ports.append(current_port)
        current_port += random.randint(100, 1000)

    return available_ports


def cleanup_processes(processes: List[mp.Process]):
    for process in processes:
        logger.info(f"Terminating process group {process.pid}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group may already be terminated
            pass

    # Wait for processes to terminate
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            logger.warning(
                f"Process {process.pid} did not terminate gracefully, forcing kill"
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    logger.info("All process groups terminated")


def main():
    # CUDA runtime isn't fork-safe, which can lead to subtle bugs or crashes
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Launch SGLang router and server processes"
    )
    # Early config flag for YAML
    parser.add_argument("--config", type=str, default=os.getenv("SGLANG_CONFIG"), help="Path to YAML config file with server/router settings")

    ServerArgs.add_cli_args(parser)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    parser.add_argument(
        "--router-dp-worker-base-port",
        type=int,
        default=31000,
        help="Base port number for data parallel workers",
    )

    # Merge YAML config (if provided) into CLI args with CLI taking precedence
    def _normalize_key(k: str) -> str:
        return k.replace("-", "_").lower()

    def _server_key_to_dest(k: str) -> str:
        k = _normalize_key(k)
        if k in ("model",):
            return "model_path"
        if k in ("tp", "tp_size", "tensor_parallel_size"):
            return "tensor_parallel_size"
        if k in ("pp", "pp_size", "pipeline_parallel_size"):
            return "pipeline_parallel_size"
        if k in ("dp", "dp_size", "data_parallel_size"):
            return "data_parallel_size"
        if k in ("ep", "ep_size", "expert_parallel_size"):
            return "expert_parallel_size"
        return k

    def _router_key_to_dest(k: str) -> str:
        k = _normalize_key(k)
        # prefix with router_
        if k.startswith("router_"):
            return k
        return f"router_{k}"

    def _option_dests_from_tokens(tokens: List[str]) -> set[str]:
        dests = set()
        opt_to_act = parser._option_string_actions  # type: ignore[attr-defined]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("--"):
                opt = tok.split("=", 1)[0]
                act = opt_to_act.get(opt)
                if act is not None:
                    dests.add(act.dest)
                i += 1
            i += 1
        return dests

    def _dest_to_flag(dest: str) -> Optional[str]:
        for act in parser._actions:  # type: ignore[attr-defined]
            if getattr(act, "dest", None) == dest:
                for opt in getattr(act, "option_strings", []):
                    if opt.startswith("--"):
                        return opt
        return None

    def _append_flag(additions: List[str], dest: str, value: Any):
        flag = _dest_to_flag(dest)
        if not flag:
            return
        action = parser._option_string_actions.get(flag)  # type: ignore[attr-defined]
        if isinstance(action, argparse._StoreTrueAction):  # type: ignore[attr-defined]
            if bool(value):
                additions.append(flag)
        elif getattr(action, "nargs", None) in ("+", "*"):
            if value is None:
                return
            additions.append(flag)
            if isinstance(value, (list, tuple)):
                additions.extend([str(x) for x in value])
            else:
                additions.append(str(value))
        else:
            if value is None:
                return
            additions.extend([flag, str(value)])

    # Extract early argv without program name
    raw_argv = sys.argv[1:]
    explicit_dests = _option_dests_from_tokens(raw_argv)
    yaml_additions: List[str] = []

    if parser.parse_known_args(raw_argv)[0].config and yaml is not None:
        cfg_path = parser.parse_known_args(raw_argv)[0].config
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level YAML must be a mapping")
        server_cfg = cfg.get("server", cfg)
        router_cfg = cfg.get("router", {})
        if not isinstance(server_cfg, dict):
            server_cfg = {}
        if not isinstance(router_cfg, dict):
            router_cfg = {}

        # Server fields
        for k, v in server_cfg.items():
            dest = _server_key_to_dest(k)
            if dest in explicit_dests:
                continue
            _append_flag(yaml_additions, dest, v)

        # Router fields (dest must include router_ prefix)
        for k, v in router_cfg.items():
            dest = _router_key_to_dest(k)
            if dest in explicit_dests:
                continue
            _append_flag(yaml_additions, dest, v)

    # Re-parse with YAML-derived additions appended (so CLI wins)
    args = parser.parse_args([*raw_argv, *yaml_additions])

    server_args = ServerArgs.from_cli_args(args)
    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

    # Find available ports for workers
    worker_ports = find_available_ports(
        args.router_dp_worker_base_port, server_args.dp_size
    )

    # Start server processes
    server_processes = []

    for i, worker_port in enumerate(worker_ports):
        logger.info(f"Launching DP server process {i} on port {worker_port}")
        proc = launch_server_process(server_args, worker_port, i)
        server_processes.append(proc)

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes(server_processes))
    signal.signal(
        signal.SIGTERM, lambda sig, frame: cleanup_processes(server_processes)
    )
    signal.signal(
        signal.SIGQUIT, lambda sig, frame: cleanup_processes(server_processes)
    )

    # Update router args with worker URLs
    router_args.worker_urls = [
        f"http://{server_args.host}:{port}" for port in worker_ports
    ]

    # Start the router
    try:
        launch_router(router_args)
    except Exception as e:
        logger.error(f"Failed to start router: {e}")
        cleanup_processes(server_processes)
        sys.exit(1)


if __name__ == "__main__":
    main()
