"""Start or reuse a warm H2O Lightning server on the Windows GTX 1650."""

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".cache/h2o-lightning-runtime/vulkan/llama-server.exe"
WEIGHTS = ROOT / ".cache/h2o-lightning-weights/h2o-lightning-4b-Q4_K_M.gguf"
MODEL = "h2oai/h2o-lightning-4b"


def ready(origin):
    http = build_opener(ProxyHandler({}))
    try:
        with http.open(origin + "/health", timeout=2) as stream:
            if json.load(stream).get("status") != "ok":
                return False
        with http.open(origin + "/v1/models", timeout=2) as stream:
            return any(model.get("id") == MODEL for model in json.load(stream).get("data", []))
    except (URLError, TimeoutError, ValueError):
        return False


def is_server(pid):
    processes = subprocess.check_output(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return any(row[:2] == [RUNTIME.name, str(pid)] for row in csv.reader(processes.splitlines()))


def start(port, state, log, origin):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Loopback port {port} is unavailable. Choose a free --port.") from exc

    with log.open("wb") as output:
        server = subprocess.Popen([
            str(RUNTIME), "--model", str(WEIGHTS), "--alias", MODEL,
            "--host", "127.0.0.1", "--port", str(port), "--device", "Vulkan1",
            "--n-gpu-layers", "99", "--fit", "off", "--ctx-size", "2048",
            "--parallel", "2", "--batch-size", "512", "--ubatch-size", "512",
            "--offline", "--no-ui", "--sleep-idle-seconds", "-1", "--cache-ram", "0",
            "--log-verbosity", "4",
        ], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and server.poll() is None:
            if ready(origin):
                if b"offloaded 34/34 layers to GPU" in log.read_bytes() and server.poll() is None:
                    state.write_text(str(server.pid), encoding="ascii")
                    return server.pid
                raise RuntimeError(f"Could not verify full GPU offload. See {log}.")
            time.sleep(.25)
        raise RuntimeError(f"H2O did not start. See {log}.")
    except BaseException:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This launcher requires Windows.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    # Windows releases the file lock automatically when the launcher exits.
    import msvcrt

    work = RUNTIME.parent.parent / f"warm-{args.port}"
    work.mkdir(parents=True, exist_ok=True)
    state = work / "server.pid"
    log = work / "server.log"
    origin = f"http://127.0.0.1:{args.port}"
    with (work / "launcher.lock").open("a+b") as lock:
        lock.seek(0)
        deadline = time.monotonic() + 65
        while True:
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise SystemExit(f"Timed out waiting for the launcher on port {args.port}.") from None
                time.sleep(.25)

        if state.is_file():
            raw_pid = state.read_text(encoding="ascii").strip()
            if raw_pid.isdecimal() and is_server(int(raw_pid)):
                if not ready(origin):
                    raise SystemExit(f"The H2O process on port {args.port} has no ready model. See {log}.")
                print(f"Reusing warm H2O GPU server at {origin} (PID {raw_pid}).")
                return

        for path in (RUNTIME, WEIGHTS):
            if not path.is_file():
                raise SystemExit(f"Local H2O asset missing: {path}")
        try:
            pid = start(args.port, state, log, origin)
        except (OSError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from None
        print(f"H2O GPU server ready at {origin} (PID {pid}, all 34 layers on GPU).")


if __name__ == "__main__":
    main()
