"""Standalone trusted C89 builder. Can also be copied into an isolated workspace."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


def discover_toolchain():
    """Resolve Apple's proxy tools before entering a workspace-only sandbox."""
    tools = {}
    for name in ["clang", "ar", "nm"]:
        result = subprocess.run(
            ["xcrun", "--find", name], capture_output=True, text=True, check=True, timeout=20
        )
        tools[name] = result.stdout.strip()
    result = subprocess.run(
        ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True, timeout=20
    )
    tools["sdkroot"] = result.stdout.strip()
    return tools


def digest_tree(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("SDK must not contain symlinks")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def invoke(argv, directory, timeout=120):
    result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv[0]}")
    return result.stdout


def build(
    project,
    framework=None,
    *,
    client=None,
    run=False,
    sdk_sha256=None,
    output="BuildFiles/Arpo",
    sanitize=False,
    compile_only=False,
):
    project = Path(project).resolve()
    framework = Path(framework or project / ".sdk").resolve()
    if sdk_sha256 and digest_tree(framework) != sdk_sha256:
        raise ValueError("SDK was changed; restore the original seed instead of modifying vendor files")
    architecture = platform.machine()
    if sys.platform != "darwin" or architecture not in {"arm64", "x86_64"}:
        raise ValueError("This Eco build profile requires native macOS clang")
    metadata = project / ".arpo-toolchain.json"
    toolchain = json.loads(metadata.read_text()) if metadata.is_file() else discover_toolchain()
    destination = (project / output).resolve()
    if not destination.is_relative_to(project) or destination == project:
        raise ValueError("Build output must be inside the project")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # SDK headers use compiler extensions: system includes isolate vendor warnings.
    # Student sources themselves are checked with strict ANSI C89.
    includes = []
    for path in sorted(project.rglob("*")):
        if path.is_dir() and path.name in {"SharedFiles", "HeaderFiles", "DependenciesFiles"}:
            if not path.is_relative_to(framework) and not path.is_relative_to(destination):
                includes += ["-I", str(path)]
    for path in sorted(framework.glob("*/SharedFiles")):
        includes += ["-isystem", str(path)]
    flags = [
        "-std=c89",
        "-pedantic-errors",
        "-Werror=implicit-function-declaration",
        "-DECO_LIB",
        "-DUGUID_UTILITY",
        "-DECO_APPLE",
        "-DECO_MAC",
        "-DECO_AARCH64" if architecture == "arm64" else "-DECO_X64",
        "-O2",
        "-isysroot",
        toolchain["sdkroot"],
        *includes,
    ]
    if sanitize:
        flags += ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"]
    sources = [
        p
        for p in sorted(project.rglob("*.c"))
        if "SourceFiles" in p.parts
        and "UnitTestFiles" not in p.parts
        and not p.is_relative_to(framework)
        and not p.is_relative_to(destination)
    ]
    if not sources:
        raise ValueError("No component sources in SourceFiles")
    objects = []
    forbidden = re.compile(
        r"^_(?:malloc|calloc|realloc|free|memcpy|memmove|memset|memcmp|strlen|strcpy|strncpy|"
        r"strcmp|strcat|printf|fprintf|sprintf|snprintf|puts|putchar|scanf|fopen|fclose|"
        r"fread|fwrite|qsort|bsearch|abs|labs|(?:sqrt|cbrt|pow|exp|log|log10|log2|sin|cos|"
        r"tan|asin|acos|atan|atan2|sinh|cosh|tanh|fabs|ceil|floor|fmod|modf|frexp|ldexp|"
        r"hypot|tgamma|lgamma)[fl]?)$"
    )
    for index, source in enumerate(sources):
        obj = destination / f"component-{index:03d}.o"
        invoke([toolchain["clang"], *flags, "-c", str(source), "-o", str(obj)], project)
        undefined = invoke([toolchain["nm"], "-u", str(obj)], project)
        blocked = [
            line.split()[-1]
            for line in undefined.splitlines()
            if line.split() and forbidden.fullmatch(line.split()[-1])
        ]
        if blocked:
            raise ValueError("Forbidden library calls in component: " + ", ".join(blocked))
        objects.append(obj)
    library = destination / "libcomponent.a"
    # ar rcs preserves obsolete members; remove only this generated archive before rebuilding.
    library.unlink(missing_ok=True)
    invoke([toolchain["ar"], "rcs", str(library), *map(str, objects)], project)
    if compile_only:
        print("C89 component scaffold compiled successfully (unimplemented symbols are allowed).", flush=True)
        return {"library": str(library), "standard": "c89", "phase": "component_compile"}
    clients = (
        [Path(client).resolve()] if client else sorted((project / "UnitTestFiles/SourceFiles").glob("*.c"))
    )
    if not clients:
        raise ValueError("A console EcoMain client is required")
    libraries = []
    for component in ["Eco.System1", "Eco.InterfaceBus1", "Eco.MemoryManager1", "Eco.FileSystemManagement1"]:
        matches = sorted(
            (framework / component / f"BuildFiles/Mac/{architecture}/StaticRelease").glob("lib*.a")
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one Mac/{architecture} static library for {component}")
        libraries += matches
    binary = destination / ("trusted-client" if client else "client")
    invoke(
        [
            toolchain["clang"],
            *flags,
            *map(str, clients),
            str(library),
            *map(str, libraries),
            "-o",
            str(binary),
        ],
        project,
    )
    record = {
        "component_sources": [str(p.relative_to(project)) for p in sources],
        "library": str(library),
        "client": str(binary),
        "architecture": architecture,
        "standard": "c89",
        "sanitizer": "undefined" if sanitize else None,
        "sdk_sha256": digest_tree(framework),
        "build_time_s": time.perf_counter() - started,
    }
    if run:
        record["client_output"] = invoke([str(binary)], project, timeout=30)
        record["client_exit_code"] = 0
    (destination / "build-result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("C89 component and EcoMain client built successfully.", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a real Eco ACOM component and its console client")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--framework", type=Path)
    parser.add_argument(
        "--client", type=Path, help="Trusted external EcoMain client, replacing student tests"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--sanitize", action="store_true", help="Check student sources for undefined behavior"
    )
    parser.add_argument("--sdk-sha256")
    parser.add_argument(
        "--compile-only", action="store_true", help="Check ABI scaffolding before AI generation"
    )
    parser.add_argument("--output", default="BuildFiles/Arpo")
    args = parser.parse_args(argv)
    try:
        build(
            args.project,
            args.framework,
            client=args.client,
            run=args.run,
            sdk_sha256=args.sdk_sha256,
            output=args.output,
            sanitize=args.sanitize,
            compile_only=args.compile_only,
        )
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print("Build error:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
