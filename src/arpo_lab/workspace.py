"""Workspace file tools and bounded subprocess execution."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def confined(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Path must be relative and must not contain '..'")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path leaves workspace (possibly through symlink)")
    # Refuse even inward symlinks to keep artifacts unambiguous.
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlinks are not allowed")
    return resolved


def sandbox_available():
    return sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()


def sandbox_profile(root, reads=(), read_only_paths=()):
    allowed = [
        str(Path(root).resolve()),
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/dev",
        "/Library/Developer",
        "/Library/Apple",
        "/Applications/Xcode.app",
        "/opt/homebrew",
        "/private/var/db",
        "/private/etc",
        sys.base_prefix,
        sys.prefix,
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "dyld-support.sb")',
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow file-read-metadata)",
    ]
    for path in dict.fromkeys(allowed):
        path = str(Path(path).resolve())
        lines.append(f"(allow file-read* (subpath {json.dumps(path)}))")
        lines.append(f"(allow file-map-executable (subpath {json.dumps(path)}))")
    for path in reads:
        lines.append(f"(allow file-read* (literal {json.dumps(str(Path(path).resolve()))}))")
    lines.append(f"(allow file-write* (subpath {json.dumps(str(Path(root).resolve()))}))")
    for relative in read_only_paths:
        path = confined(root, relative)
        lines.append(f"(deny file-write* (subpath {json.dumps(str(path))}))")
    lines.append('(allow file-write* (literal "/dev/null"))')
    return "\n".join(lines)


def execute(argv, root, config, *, cwd=".", timeout=None, extra_reads=()):
    if not isinstance(argv, list) or not argv or not all(isinstance(s, str) and s for s in argv):
        raise ValueError("argv must be a nonempty list of nonempty strings")
    root = Path(root).resolve()
    directory = confined(root, cwd)
    settings = config["tools"]
    mode = settings["sandbox"]
    available = sandbox_available()
    if mode == "required" and not available:
        raise RuntimeError("macOS sandbox-exec unavailable; required sandbox cannot run")
    sandboxed = mode != "off" and available
    command = argv
    if sandboxed:
        command = [
            "/usr/bin/sandbox-exec",
            "-p",
            sandbox_profile(root, extra_reads, settings["read_only_paths"]),
            *argv,
        ]
    temporary = root / ".arpo-tmp"
    temporary.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        "TMPDIR": str(temporary),
        "LANG": "en_US.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CLANG_MODULE_CACHE_PATH": str(temporary / "clang-cache"),
        "XCRUN_CACHE_DISABLED": "1",
        **settings["environment"],
    }
    started = time.perf_counter()
    with tempfile.TemporaryFile(dir=temporary) as output:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            code = process.wait(timeout=timeout or settings["timeout_s"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            code, timed_out = -9, True
        size = output.tell()
        output.seek(0)
        text = output.read(settings["max_output_chars"] * 4).decode("utf-8", errors="replace")
    return {
        "argv": argv,
        "exit_code": code,
        "timed_out": timed_out,
        "sandboxed": sandboxed,
        "duration_s": time.perf_counter() - started,
        "output": text[: settings["max_output_chars"]],
        "output_truncated": size > len(text.encode("utf-8")) or len(text) > settings["max_output_chars"],
    }


class Workspace:
    def __init__(self, root, config):
        self.root, self.config = Path(root).resolve(), config

    def action(self, action, remaining_s):
        settings = self.config["tools"]
        kind = action.get("type")
        if kind == "invalid":
            raise ValueError(action.get("error", "Invalid tool call"))
        if kind == "run":
            if not settings["allow_run"]:
                raise ValueError("Agent command execution is disabled")
            argv = action.get("argv")
            if not isinstance(argv, list) or not argv or argv[0] not in settings["allowed_executables"]:
                raise ValueError(
                    "Executable is not in tools.allowed_executables; use its exact configured name"
                )
            return execute(
                argv,
                self.root,
                self.config,
                cwd=action.get("cwd", "."),
                timeout=min(settings["timeout_s"], max(0.1, remaining_s)),
            )
        path = confined(self.root, action.get("path", "."))
        if kind == "write":
            if any(
                path == confined(self.root, p) or path.is_relative_to(confined(self.root, p))
                for p in settings["read_only_paths"]
            ):
                raise ValueError("This path is read-only: " + action["path"])
            content = action.get("content")
            if not isinstance(content, str):
                raise ValueError("write.content must be a string")
            encoded = content.encode("utf-8")
            if len(encoded) > settings["max_file_bytes"]:
                raise ValueError("File exceeds max_file_bytes")
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and not p.is_symlink())
            old_size = path.stat().st_size if path.is_file() else 0
            if used - old_size + len(encoded) > settings["max_workspace_bytes"]:
                raise ValueError("Workspace exceeds max_workspace_bytes")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
            return {"written": path.relative_to(self.root).as_posix(), "bytes": len(encoded)}
        if kind == "read":
            with path.open("rb") as stream:
                content = stream.read(settings["max_file_bytes"] + 1)
            if len(content) > settings["max_file_bytes"]:
                raise ValueError("File exceeds max_file_bytes")
            return {"path": path.relative_to(self.root).as_posix(), "content": content.decode("utf-8")}
        if kind == "list":
            entries = []
            for item in sorted(path.iterdir()):
                if len(entries) >= settings["max_list_entries"]:
                    break
                if item.name == ".arpo-tmp":
                    continue
                entries.append(
                    {
                        "path": item.relative_to(self.root).as_posix(),
                        "kind": "symlink" if item.is_symlink() else "dir" if item.is_dir() else "file",
                    }
                )
            return {"entries": entries}
        raise ValueError(f"Unknown action type: {kind}")


def copy_seed(seed, destination):
    if seed:
        shutil.copytree(seed, destination, dirs_exist_ok=True)
