"""Own only the MLX server started by this invocation; preserve existing servers."""

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from .api import Client


def check_model(config):
    data = Client(config).models()
    ids = {row.get("id") for row in data.get("data", [])}
    name = config["model"]["name"]
    expected = {name}
    # MLX advertises resolved local paths, including targets of external-SSD symlinks.
    if Path(name).is_dir():
        expected.add(str(Path(name).resolve()))
    if not ids.intersection(expected):
        raise ValueError(
            "Running API does not advertise the configured model; stop it or select its model ID"
        )


@contextmanager
def local_server(config, repo, enabled=True, progress=print):
    try:
        check_model(config)
    except ValueError:
        raise
    except OSError:
        if not enabled:
            raise RuntimeError(
                "API is unavailable; start arpo serve or allow automatic server startup"
            ) from None
    else:
        progress("Используется уже запущенный сервер модели.")
        yield
        return
    url = urlparse(config["model"]["base_url"])
    if url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.scheme != "http" or url.path != "/v1":
        raise ValueError("Automatic server requires a local http://host:port/v1 endpoint")
    model = Path(config["model"]["name"])
    if model.is_symlink() and not model.exists():
        raise ValueError("Model directory points to unavailable storage; connect the external SSD: "
                         + str(model.readlink()))
    if not (model / "arpo-model.json").is_file():
        raise ValueError("Local model is not downloaded; run bash scripts/bootstrap.sh first")
    repo = Path(repo).resolve()
    logs = repo / ".cache"
    logs.mkdir(exist_ok=True)
    log_path = logs / ("mlx-" + time.strftime("%Y%m%d-%H%M%S") + ".log")
    command = [
        sys.executable,
        "-m",
        "arpo_lab",
        "--model",
        config["model"]["name"],
        "serve",
        "--port",
        str(url.port or 8080),
        "--host",
        url.hostname,
    ]
    progress("Запуск MLX; журнал: " + str(log_path))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 180
            last_update = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("MLX exited during startup; see " + str(log_path))
                try:
                    check_model(config)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("MLX startup timed out; see " + str(log_path)) from None
                    if time.monotonic() - last_update > 20:
                        progress("Ожидаю готовности MLX; журнал: " + str(log_path))
                        last_update = time.monotonic()
                    time.sleep(1)
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
