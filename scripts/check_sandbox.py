"""Manual macOS integration check: allow compilation, deny private files/network."""

import sys
import tempfile
from pathlib import Path

from arpo_lab.config import load_config
from arpo_lab.workspace import execute


def main():
    config = load_config(overrides={"tools": {"sandbox": "required"}})
    with tempfile.TemporaryDirectory(prefix="arpo-sandbox-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        secret = root / "private.txt"
        secret.write_text("must not be readable")
        (workspace / "test.c").write_text("int main(void) { return 0; }\n")
        python_result = execute([sys.executable, "-c", "print('python started')"], workspace, config)
        assert python_result["exit_code"] == 0, python_result
        compile_result = execute(
            ["clang", "-std=c89", "-pedantic-errors", "test.c", "-o", "test"], workspace, config
        )
        if compile_result["exit_code"]:
            print(compile_result)
            return 1
        assert execute([str(workspace / "test")], workspace, config)["exit_code"] == 0
        read_result = execute(
            [sys.executable, "-c", f"from pathlib import Path; print(Path({str(secret)!r}).read_text())"],
            workspace,
            config,
        )
        assert read_result["exit_code"] != 0, "Private file read was permitted"
        assert "PermissionError" in read_result["output"], read_result
        network_result = execute(
            [sys.executable, "-c", "import socket; socket.socket().bind(('127.0.0.1', 0))"], workspace, config
        )
        assert network_result["exit_code"] != 0, "Network was permitted"
        assert "PermissionError" in network_result["output"], network_result
    print("PASS sandbox: C89 builds and executes; private reads and networking denied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
