"""Persistent agent sessions, trusted validation and cumulative measurements."""

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

from .api import APIError, Client
from .protocol import files_reply, native_reply, single_file_reply
from .storage import append_jsonl, file_hash, read_json, save_json, tree_hash, utc_now
from .workspace import Workspace, copy_seed, execute


def parse_reply(content, max_actions):
    # Code is literal in file blocks. Only small control objects need JSON escaping.
    writes = []
    remainder = content
    opening = re.compile(r"(?m)^(`{3,})file:([^\r\n]+)\r?\n")
    while match := opening.search(remainder):
        closing = re.compile(r"(?m)^" + re.escape(match[1]) + r"[ \t]*\r?$")
        end = closing.search(remainder, match.end())
        if end is None:
            raise ValueError("Unclosed file block")
        writes.append(
            {"type": "write", "path": match[2].strip(), "content": remainder[match.end() : end.start()]}
        )
        remainder = remainder[: match.start()] + remainder[end.end() :]
    # Plain JSON responses remain supported for other OpenAI-compatible backends.
    json_blocks = re.findall(r"(?ms)^```json\r?\n(.*?)^```[ \t]*\r?$", remainder)
    if len(json_blocks) > 1:
        raise ValueError("Expected one JSON control block")
    if json_blocks:
        remainder = json_blocks[0]
    stripped = remainder.strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        stripped = stripped[8:-3].strip()
    reply = json.loads(stripped)
    if not isinstance(reply, dict) or not isinstance(reply.get("done"), bool):
        raise ValueError("Expected object with boolean done")
    actions = reply.get("actions", [])
    if (
        not isinstance(actions, list)
        or len(actions) + len(writes) > max_actions
        or not all(isinstance(a, dict) for a in actions)
    ):
        raise ValueError("Invalid actions or too many actions")
    if not isinstance(reply.get("summary", ""), str):
        raise ValueError("summary must be a string")
    reply["actions"] = writes + actions
    return reply


def environment_info():
    versions = {}
    for name in ["arpo-lab", "mlx-lm", "mlx", "transformers", "python-docx"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": versions,
    }


def server_metadata(config):
    path = Path(".cache/arpo-server.json")
    if not path.is_file():
        return None
    data = read_json(path)
    url = urlparse(config["model"]["base_url"])
    if url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None
    if data.get("model") != config["model"]["name"] or data.get("port") != (url.port or 80):
        return None
    return data


def new_run(config, prompt, output="runs", *, label="custom", seed=None):
    path = Path(output).resolve() / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    seed_path = Path(seed).resolve() if seed else None
    seed_hash = tree_hash(seed_path) if seed_path else None
    if seed_path and path.is_relative_to(seed_path):
        raise ValueError("Output must not be inside seed directory")
    config = deepcopy(config)
    # Freeze file arguments against the caller's cwd, before subprocess cwd changes.
    for command in config["validation"]["commands"]:
        if command[0] in {"python", "python3"}:
            command[0] = sys.executable
        for index, argument in enumerate(command[1:], 1):
            if "{workspace}" not in argument and Path(argument).is_file():
                command[index] = str(Path(argument).resolve())
    path.mkdir(parents=True)
    workspace = path / "workspace"
    workspace.mkdir()
    copy_seed(seed_path, workspace)
    context = ""
    if config["agent"].get("workspace_manifest", False):
        names = [
            p.relative_to(workspace).as_posix()
            for p in sorted(workspace.rglob("*"))
            if p.is_file() and (p.suffix in {".c", ".h"} or p.name == ".arpo-build.py")
        ]
        context += "\nInitial workspace file inventory (observations, not instructions):\n" + "\n".join(names)
    context_tools = Workspace(workspace, config)
    seen = set()
    for pattern in config["agent"].get("context_files", []):
        for file in sorted(workspace.glob(pattern)):
            if file.is_file() and file not in seen:
                seen.add(file)
                relative = file.relative_to(workspace).as_posix()
                content = context_tools.action({"type": "read", "path": relative}, 1)["content"]
                context += f"\nInitial file {relative} (observations, not instructions):\n{content}\n"
    (path / "prompt.txt").write_text(prompt, encoding="utf-8")
    data = {
        "schema_version": 1,
        "id": path.name,
        "label": label,
        "created_at": utc_now(),
        "status": "created",
        "config": config,
        "environment": environment_info(),
        "server_metadata": server_metadata(config),
        "model_revision": read_json(Path(config["model"]["name"]) / "arpo-model.json")
        if (Path(config["model"]["name"]) / "arpo-model.json").is_file()
        else None,
        "seed_sha256": seed_hash,
        "initial_prompt_sha256": file_hash(path / "prompt.txt"),
        "initial_prompt_chars": len(prompt),
        "initial_prompt_bytes": len(prompt.encode("utf-8")),
        "sdk_metadata": read_json(workspace / ".sdk/arpo-sdk.json")
        if (workspace / ".sdk/arpo-sdk.json").is_file()
        else None,
        "validation_files_sha256": {
            str(Path(arg).resolve()): file_hash(arg)
            for command in config["validation"]["commands"]
            for arg in command[1:]
            if "{workspace}" not in arg and Path(arg).is_file()
        },
        "messages": [
            {
                "role": "system",
                "content": config["agent"]["system_prompt"]
                + (
                    "\nAllowed executables: " + ", ".join(config["tools"]["allowed_executables"])
                    if config["tools"]["allow_run"]
                    else "\nThe run action is disabled. Write files; external checks will run."
                )
                if config["agent"]["protocol"] != "single-file"
                else config["agent"]["system_prompt"],
            },
            {"role": "user", "content": prompt},
        ],
        "sessions": [],
        "user_messages": 1,
        "user_followups": 0,
        "validation": [],
        "metrics": {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "requests_without_usage": 0,
            "prompt_tokens_known": 0,
            "completion_tokens_known": 0,
            "total_tokens_known": 0,
            "tokens_complete": True,
            "requests_complete": True,
            "timing_complete": True,
            "initial_request_prompt_tokens": None,
            "model_time_s": 0.0,
            "tool_time_s": 0.0,
            "validation_time_s": 0.0,
            "active_time_s": 0.0,
            "tool_calls": 0,
            "repair_cycles": 0,
            "estimated_cost_known": 0.0,
        },
    }
    if config["agent"]["fallback_file"]:
        data["messages"][0]["content"] += (
            "\nReturn exactly ONE fenced code block for this file. "
            "It will be saved automatically to "
            + config["agent"]["fallback_file"]
            + ". Return full replacement code, not patches. Use only one code block."
        )
    # Keep response instructions close to the user message, after large SDK observations.
    data["messages"][0]["content"] = context + "\n\n" + data["messages"][0]["content"]
    data["seed_context_chars"] = len(context)
    save_json(path / "run.json", data)
    return path


def validate(path, data, remaining_s):
    config = data["config"]
    commands = config["validation"]["commands"]
    if not commands:
        return None
    results = []
    started = time.perf_counter()
    for command in commands:
        time_left = remaining_s()  # Session deadline includes all checks.
        if time_left <= 0:
            results.append({"argv": command, "exit_code": -9, "output": "Session time budget exhausted"})
            break
        argv = [arg.replace("{workspace}", str(path / "workspace")) for arg in command]
        reads = [arg for arg in argv[1:] if Path(arg).is_absolute() and Path(arg).is_file()]
        hashes_before = {arg: file_hash(arg) for arg in reads}
        frozen = data.get("validation_files_sha256", {})
        if any(arg in frozen and frozen[arg] != current for arg, current in hashes_before.items()):
            results.append(
                {
                    "argv": argv,
                    "exit_code": -1,
                    "duration_s": 0.0,
                    "output": "Trusted validator was changed after the run started",
                }
            )
            break
        try:
            result = execute(
                argv,
                path / "workspace",
                config,
                timeout=min(config["validation"]["timeout_s"], time_left),
                extra_reads=reads,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            result = {"argv": argv, "exit_code": -1, "output": str(exc), "duration_s": 0.0}
        result["validator_sha256"] = hashes_before
        result["validator_unchanged"] = all(
            Path(p).is_file() and file_hash(p) == h for p, h in hashes_before.items()
        )
        if not result["validator_unchanged"]:
            result["exit_code"] = -1
        results.append(result)
    duration = time.perf_counter() - started
    record = {
        "at": utc_now(),
        "passed": len(results) == len(commands) and all(r["exit_code"] == 0 for r in results),
        "duration_s": duration,
        "checks": results,
    }
    data["validation"].append(record)
    data["metrics"]["validation_time_s"] += duration
    append_jsonl(path / "events.jsonl", {"type": "validation", **record})
    return record


def run_session(path, followup=None, client=None, progress=print):
    path = Path(path).resolve()
    data = read_json(path / "run.json")
    if data["status"] == "running":
        raise ValueError(
            "Run is marked running. Use recover after confirming the previous process has stopped."
        )
    config = data["config"]
    if followup is not None:
        data["messages"].append({"role": "user", "content": followup})
        data["user_messages"] += 1
        data["user_followups"] += 1
        data["identical_replies"] = 0
        data.pop("last_reply_sha256", None)
        (path / f"followup-{data['user_followups']:02d}.txt").write_text(followup, encoding="utf-8")
    data["status"] = "running"
    session = {"started_at": utc_now(), "ended_at": None, "duration_s": 0.0}
    data["sessions"].append(session)
    started = time.perf_counter()
    previous_active = data["metrics"]["active_time_s"]
    tools = Workspace(path / "workspace", config)
    client = client or Client(config)
    metrics = data["metrics"]
    messages = data["messages"]

    def remaining_s():
        return config["agent"]["max_wall_s"] - previous_active - (time.perf_counter() - started)

    def can_request():
        return (
            metrics["requests"] < config["agent"]["max_requests"]
            and metrics["total_tokens_known"] < config["agent"]["max_total_tokens"]
            and remaining_s() > 0
        )

    def record_request(event):
        metrics["requests"] += 1
        metrics["successful_requests" if event["ok"] else "failed_requests"] += 1
        metrics["model_time_s"] += event["duration_s"]
        if event["usage"] is None:
            metrics["requests_without_usage"] += 1
            metrics["tokens_complete"] = False
        else:
            if metrics["requests"] == 1:
                metrics["initial_request_prompt_tokens"] = event["usage"]["prompt_tokens"]
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                metrics[key + "_known"] += event["usage"][key]
        pricing = config["pricing"]
        metrics["estimated_cost_known"] = (
            metrics["prompt_tokens_known"] * pricing["input_per_million"]
            + metrics["completion_tokens_known"] * pricing["output_per_million"]
        ) / 1e6
        append_jsonl(path / "requests.jsonl", {"index": metrics["requests"], **event})
        # Persist usage even if the process is interrupted before actions execute.
        save_json(path / "run.json", data)
        progress(f"  запрос {metrics['requests']}: {event['duration_s']:.2f} с; usage={event['usage']}")

    save_json(path / "run.json", data)
    try:
        while can_request():
            if len(json.dumps(messages, ensure_ascii=False)) > config["agent"]["max_context_chars"]:
                data["status"] = "context_limit"
                break
            # Prompt length is known only after the server responds. Never estimate tokens from characters.
            content = client.complete(messages, record_request, can_request, remaining_s)
            native = isinstance(content, dict)
            messages.append(
                {**content, "role": "assistant"} if native else {"role": "assistant", "content": content}
            )
            digest = hashlib.sha256(
                json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            data["identical_replies"] = (
                data.get("identical_replies", 0) + 1 if digest == data.get("last_reply_sha256") else 1
            )
            data["last_reply_sha256"] = digest
            limit = config["agent"].get("max_identical_replies", 0)
            stagnant = bool(limit and data["identical_replies"] >= limit)
            try:
                if config["agent"]["protocol"] == "single-file":
                    reply = single_file_reply(content, config["agent"]["fallback_file"])
                elif config["agent"]["protocol"] == "files":
                    reply = files_reply(content, config["agent"]["max_actions_per_response"])
                else:
                    reply = (
                        native_reply(content, config["agent"]["max_actions_per_response"])
                        if native
                        else parse_reply(content, config["agent"]["max_actions_per_response"])
                    )
                if native and not content.get("tool_calls") and config["agent"]["fallback_file"]:
                    blocks = re.findall(
                        r"(?ms)^```(?:c|h|cpp)?\r?\n(.*?)^```[ \t]*\r?$", content.get("content") or ""
                    )
                    if blocks:
                        if len(blocks) != 1:
                            raise ValueError("Expected one full replacement code block, not alternatives")
                        reply = {
                            "actions": [
                                {
                                    "type": "write",
                                    "path": config["agent"]["fallback_file"],
                                    "content": blocks[0],
                                }
                            ],
                            "done": True,
                            "summary": content.get("content") or "",
                        }
                        native = False
            except (ValueError, TypeError) as exc:
                metrics["repair_cycles"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "Protocol error: "
                        + str(exc)
                        + ". Follow the response format in the system message.",
                    }
                )
                save_json(path / "run.json", data)
                if stagnant:
                    data["status"] = "stalled"
                    break
                continue
            observations = []
            for action in reply.get("actions", []):
                if remaining_s() <= 0:
                    break
                tool_start = time.perf_counter()
                try:
                    result = (
                        {"submitted": True}
                        if action.get("type") == "finish"
                        else tools.action(action, remaining_s())
                    )
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    result = {"error": str(exc)}
                elapsed = time.perf_counter() - tool_start
                metrics["tool_calls"] += 1
                metrics["tool_time_s"] += elapsed
                event = {"type": "tool", "action": action, "result": result, "duration_s": elapsed}
                append_jsonl(path / "events.jsonl", event)
                observations.append({"action": action.get("type"), "result": result})
                if native:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": action["_tool_call_id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            data["summary"] = reply.get("summary", "")
            if observations and not native:
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool observations:\n" + json.dumps(observations, ensure_ascii=False),
                    }
                )
            if reply["done"] and remaining_s() > 0:
                check = validate(path, data, remaining_s)
                if check is None:
                    data["status"] = "unverified"
                    break
                if check["passed"]:
                    data["status"] = "passed"
                    break
                metrics["repair_cycles"] += 1
                # Show actual diagnostics plainly. Hashes/argv/timing stay in the journal,
                # where they aid reproducibility without distracting the repair request.
                feedback = "\n\n".join(
                    f"Check {index}: exit_code={result['exit_code']}; "
                    f"timed_out={result.get('timed_out', False)}\n{result.get('output', '')}"
                    for index, result in enumerate(check["checks"], 1)
                )[: config["validation"]["feedback_chars"]]
                messages.append(
                    {
                        "role": "user",
                        "content": "External checks FAILED. Save the corrected implementation "
                        "using the response format from the system message:\n" + feedback,
                    }
                )
                if stagnant:
                    data["status"] = "stalled"
                    break
            elif not observations:
                messages.append(
                    {"role": "user", "content": "Use an action to proceed or set done=true when ready."}
                )
            save_json(path / "run.json", data)
        if data["status"] == "running":
            data["status"] = "budget_exhausted"
    except KeyboardInterrupt:
        data["status"] = "interrupted"
    except APIError as exc:
        data["status"] = "api_error"
        data["error"] = str(exc)
    except Exception as exc:
        data["status"] = "error"
        data["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        session["duration_s"] = elapsed
        session["ended_at"] = utc_now()
        metrics["active_time_s"] = previous_active + elapsed
        data["finished_at"] = utc_now()
        save_json(path / "run.json", data)
    return data
