import time
import uuid
from copy import deepcopy
from pathlib import Path

from .agent import new_run, run_session
from .api import Client
from .storage import append_jsonl, file_hash, save_json, utc_now


def experiment(config, short, detailed, output="runs", seed=None, progress=print):
    path = Path(output).resolve() / (
        "experiment-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    path.mkdir(parents=True)
    texts = {
        "short": Path(short).read_text(encoding="utf-8"),
        "detailed": Path(detailed).read_text(encoding="utf-8"),
    }
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config": deepcopy(config),
        "prompts_sha256": {"short": file_hash(short), "detailed": file_hash(detailed)},
        "runs": [],
    }
    save_json(path / "experiment.json", manifest)
    if config["experiment"]["warmup"]:
        progress("Прогрев модели (сохраняется отдельно и не входит в метрики решений).")
        warm_config = deepcopy(config)
        warm_config["model"]["max_tokens"] = 8
        Client(warm_config).complete(
            [{"role": "user", "content": "Reply OK."}],
            lambda event: append_jsonl(path / "warmup.jsonl", event),
        )
    for repeat in range(config["experiment"]["repeats"]):
        order = ["short", "detailed"]
        if config["experiment"]["order"] == "detailed-first" or (
            config["experiment"]["order"] == "alternate" and repeat % 2
        ):
            order.reverse()
        for label in order:
            progress(f"Повтор {repeat + 1}, {label}")
            run = new_run(config, texts[label], path, label=label, seed=seed)
            manifest["runs"].append({"path": run.name, "label": label, "repeat": repeat + 1})
            save_json(path / "experiment.json", manifest)
            result = run_session(run, progress=progress)
            progress(f"  результат: {result['status']}; {run}")
            if result["status"] == "interrupted":
                manifest["interrupted"] = True
                save_json(path / "experiment.json", manifest)
                return path
    manifest["finished_at"] = utc_now()
    save_json(path / "experiment.json", manifest)
    return path
