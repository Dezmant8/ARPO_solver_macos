"""Import a user-supplied Eco distribution without redistributing proprietary files."""

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .eco_build import build, discover_toolchain
from .storage import file_hash, read_json, save_json, tree_hash, utc_now


def require_apple_silicon():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError(
            "Automatic MLX setup requires Apple Silicon macOS; Intel needs a separate API server"
        )
    if not shutil.which("clang"):
        raise ValueError("Install Apple Command Line Tools: xcode-select --install; then rerun setup")


def select_archive(source, pattern):
    matches = sorted(Path(source).glob(pattern), reverse=True)
    if not matches:
        raise ValueError(f"Missing {pattern} in {source}")
    return matches[0]


def checked_member(info):
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or "\\" in info.filename or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe archive member: {info.filename}")
    if info.file_size > 300_000_000:
        raise ValueError("Archive member exceeds 300 MB")
    return path


def extract_selected(archive, destination, select):
    with zipfile.ZipFile(archive) as stream:
        infos = stream.infolist()
        if sum(i.file_size for i in infos) > 2_000_000_000:
            raise ValueError("Archive exceeds 2 GB unpacked")
        for info in infos:
            path = checked_member(info)
            if not info.is_dir() and select(path):
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.read(info))


def setup_eco(source=None, repo=Path("."), progress=print):
    require_apple_silicon()
    repo = Path(repo).resolve()
    source = Path(source or Path.home() / "Downloads/Design operating systems").expanduser().resolve()
    framework_archive = select_archive(source, "Eco.Framework*.zip")
    candidates = sorted(p for p in source.glob("*wizard*macos*arm64*.zip") if "vscode" not in p.name)
    if not candidates:
        raise ValueError("Missing eco-wizard macos arm64 ZIP in " + str(source))
    wizard_archive = candidates[-1]
    framework_hash, wizard_hash = file_hash(framework_archive), file_hash(wizard_archive)
    home = repo / ".eco"
    home.mkdir(exist_ok=True)
    sdk = home / ("sdk-" + framework_hash[:12]) / "Eco.Framework"
    metadata = sdk / "arpo-sdk.json"
    if not metadata.exists():
        sdk.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="eco-import-", dir=home) as folder:
            staging = Path(folder)
            extract_selected(
                framework_archive,
                staging,
                lambda p: (
                    p.parts[0] == "Eco.Framework"
                    and ("SharedFiles" in p.parts or "/BuildFiles/Mac/arm64/" in str(p))
                    and not p.name.startswith("._")
                ),
            )
            incoming = staging / "Eco.Framework"
            if not (incoming / "Eco.Core1/SharedFiles/IEcoBase1.h").is_file():
                raise ValueError("Archive does not contain the Eco.Core1 development kit")
            files = {
                p.relative_to(incoming).as_posix(): file_hash(p) for p in incoming.rglob("*") if p.is_file()
            }
            save_json(
                incoming / "arpo-sdk.json",
                {
                    "archive": framework_archive.name,
                    "archive_sha256": framework_hash,
                    "architecture": "arm64",
                    "files": files,
                },
            )
            if sdk.exists():
                raise ValueError(
                    "Incomplete prior Eco import at " + str(sdk) + "; choose another repo folder"
                )
            incoming.rename(sdk)
    info = read_json(metadata)
    for name, expected in info["files"].items():
        path = sdk / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Installed SDK has changed: " + str(path))
    binary = home / "bin" / ("eco-wizard-" + wizard_hash[:12])
    binary.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(wizard_archive) as stream:
        choices = [i for i in stream.infolist() if checked_member(i).name == "eco-wizard" and not i.is_dir()]
        if len(choices) != 1:
            raise ValueError("Expected one eco-wizard executable in the archive")
        content = stream.read(choices[0])
    if not binary.exists():
        binary.write_bytes(content)
        binary.chmod(0o755)
    elif binary.read_bytes() != content:
        raise ValueError("Installed eco-wizard was changed")
    version = subprocess.run(
        [str(binary), "version", "--json"], capture_output=True, text=True, check=True, timeout=20
    )
    install = {
        "source": str(source),
        "sdk": str(sdk),
        "sdk_sha256": tree_hash(sdk),
        "framework_archive": framework_archive.name,
        "framework_archive_sha256": framework_hash,
        "wizard": str(binary),
        "wizard_archive_sha256": wizard_hash,
        "wizard_version": json.loads(version.stdout)["version"],
        "prepared_at": utc_now(),
    }
    save_json(home / "install.json", install)
    progress(f"Eco Framework: {sdk}\neco-wizard: {install['wizard_version']}")
    return install


def installed(repo=Path(".")):
    path = Path(repo).resolve() / ".eco/install.json"
    if not path.exists():
        raise ValueError("Eco is not prepared. Run bash scripts/bootstrap.sh --eco-dir <directory>")
    data = read_json(path)
    if tree_hash(Path(data["sdk"])) != data["sdk_sha256"]:
        raise ValueError("SDK fingerprint differs from the setup record")
    return data


def generate_project(parent, name, install):
    if not name.startswith("Eco.") or not all(c.isalnum() or c in "._" for c in name):
        raise ValueError("Project name must start with Eco. and contain letters, digits, dots or underscores")
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    command = [
        install["wizard"],
        "new",
        "-o",
        str(parent),
        "-n",
        name,
        "-t",
        "COM",
        "-l",
        "C",
        "--opt",
        "ut,pn",
        "--env",
        "true",
        "--no-design-docs",
        "--if-exists",
        "fail",
        "--format",
        "json",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ECO_FRAMEWORK": install["sdk"]},
    )
    if result.returncode:
        raise RuntimeError("eco-wizard failed: " + result.stdout[-3000:] + result.stderr[-1000:])
    receipt = json.loads(result.stdout)
    project = Path(receipt["project_dir"])
    if not project.is_relative_to(parent):
        raise ValueError("eco-wizard returned a project outside its output directory")
    shutil.copytree(install["sdk"], project / ".sdk")
    shutil.copyfile(Path(__file__).with_name("eco_build.py"), project / ".arpo-build.py")
    save_json(project / ".arpo-toolchain.json", discover_toolchain())
    # Fix vendor C11 defaults and stale library file references for optional Xcode use.
    for path in project.glob("AssemblyFiles/Mac/**/*.pbxproj"):
        text = path.read_text(encoding="utf-8-sig")
        text = text.replace("GCC_C_LANGUAGE_STANDARD = gnu11;", "GCC_C_LANGUAGE_STANDARD = c89;")
        text = text.replace("arm64 x86_64/Debug", "arm64/StaticRelease")
        text = text.replace("../../../../Eco.", "../../../.sdk/Eco.")
        path.write_text(text, encoding="utf-8")
    save_json(
        project / "arpo-scaffold.json",
        {
            "wizard_version": install["wizard_version"],
            "sdk_archive": install["framework_archive"],
            "sdk_sha256": install["sdk_sha256"],
            "name": name,
            "receipt": receipt,
        },
    )
    return project


def smoke_test(install, repo=Path(".")):
    home = Path(repo).resolve() / ".eco"
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=home) as folder:
        project = generate_project(folder, "Eco.ArpoSmoke", install)
        result = build(project, run=True, sdk_sha256=install["sdk_sha256"])
    save_json(
        home / "smoke.json",
        {
            "at": utc_now(),
            "sdk_sha256": install["sdk_sha256"],
            "status": "passed",
            "standard": "c89",
            "client_exit_code": result["client_exit_code"],
        },
    )
    print("Eco smoke check passed: C89 build, real component factory, bus and memory allocator.")
