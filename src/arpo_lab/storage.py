import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Symlinks are not supported in seed: {item}")
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()
