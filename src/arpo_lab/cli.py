import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from .agent import compact_history, environment_info, new_run, run_session
from .api import Client
from .config import increase_run_budgets, load_config
from .eco import installed, setup_eco, smoke_test
from .experiment import experiment
from .lab import init_lab, make_lab_reports, resume_lab, run_lab
from .report import load_runs, write_report
from .storage import read_json, save_json, utc_now
from .workspace import sandbox_available


def parser():
    p = argparse.ArgumentParser(description="Локальный ИИ агент и сравнение промптов для лабораторных C89")
    p.add_argument("--config", type=Path, help="TOML с настройками (до подкоманды)")
    p.add_argument("--model", help="Переопределить model.name (до подкоманды)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Проверить окружение и доступность API")
    eco_setup = sub.add_parser("eco-setup", help="Подключить SDK и eco-wizard из папки преподавателя")
    eco_setup.add_argument("--source-dir", type=Path)
    eco_setup.add_argument("--skip-smoke", action="store_true")
    sub.add_parser("eco-doctor", help="Реальная сборка и запуск шаблона ACOM на C89")
    lab_init = sub.add_parser("lab-init", help="Подготовить независимые серии для лабораторной ACOM")
    lab_init.add_argument("--directory", type=Path, required=True)
    lab_init.add_argument("--topic", required=True)
    lab_init.add_argument("--name", default="Eco.Lab1")
    lab_init.add_argument("--example", choices=["abs"])
    lab_run = sub.add_parser("lab-run", help="Запустить MLX, две серии и единый Word отчёт")
    lab_run.add_argument("directory", type=Path)
    lab_run.add_argument("--repeats", type=int)
    lab_run.add_argument("--no-server", action="store_true")
    lab_resume = sub.add_parser("lab-resume", help="Уточнить вариант с автоматическим запуском MLX")
    lab_resume.add_argument("directory", type=Path)
    lab_resume.add_argument("--run", type=Path, required=True)
    followup = lab_resume.add_mutually_exclusive_group(required=True)
    followup.add_argument("--prompt")
    followup.add_argument("--prompt-file", type=Path)
    lab_resume.add_argument("--add-requests", type=int, default=0)
    lab_resume.add_argument("--add-tokens", type=int, default=0)
    lab_resume.add_argument("--add-seconds", type=float, default=0)
    lab_resume.add_argument("--add-context-chars", type=int, default=0,
                            help="Явно увеличить предел истории без её удаления")
    lab_resume.add_argument("--compact-history", action="store_true",
                            help="Архивировать предыдущие ответы без сброса метрик")
    lab_resume.add_argument("--no-server", action="store_true")
    lab_report = sub.add_parser("lab-report", help="Пересобрать единый отчёт после заполнения пояснений")
    lab_report.add_argument("directory", type=Path)
    lab_report.add_argument("--experiment", type=Path)
    serve = sub.add_parser("serve", help="MLX сервер; держать открытым в отдельном терминале")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    serve.add_argument("extra", nargs=argparse.REMAINDER, help="Дополнительные параметры MLX после --")
    download = sub.add_parser("download", help="Скачать MLX веса и зафиксировать ревизию Hugging Face")
    download.add_argument("--revision", default="main")
    download.add_argument("--destination", type=Path, default=Path("models/qwen-coder"))
    sub.add_parser("warmup", help="Короткий запрос для прогрева, без экспериментальных метрик")
    run = sub.add_parser("run", help="Начать отдельный запуск")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Текст задачи")
    group.add_argument("--prompt-file", type=Path)
    run.add_argument("--label", default="custom")
    run.add_argument("--seed-dir", type=Path, help="Начальные файлы, например Eco SDK и шаблон")
    run.add_argument("--output", type=Path, default=Path("runs"))
    resume = sub.add_parser("resume", help="Уточнить задачу в том же запуске с накоплением метрик")
    resume.add_argument("run", type=Path)
    group = resume.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    resume.add_argument("--add-requests", type=int, default=0, help="Явно увеличить общий бюджет запросов")
    resume.add_argument("--add-tokens", type=int, default=0)
    resume.add_argument("--add-seconds", type=float, default=0)
    resume.add_argument("--add-context-chars", type=int, default=0)
    resume.add_argument("--compact-history", action="store_true")
    recover = sub.add_parser("recover", help="Снять running после аварийного завершения процесса")
    recover.add_argument("run", type=Path)
    exp = sub.add_parser("experiment", help="Парные запуски с независимыми каталогами")
    exp.add_argument("--short", type=Path, required=True)
    exp.add_argument("--detailed", type=Path, required=True)
    exp.add_argument("--repeats", type=int)
    exp.add_argument("--seed-dir", type=Path)
    exp.add_argument("--output", type=Path, default=Path("runs"))
    report = sub.add_parser("report", help="Сравнение запусков и шаблон лабораторного отчёта")
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--output", type=Path, default=Path("reports"))
    report.add_argument("--formats", default="md,html,csv,json", help="md,html,csv,json,docx через запятую")
    report.add_argument("--notes", type=Path, help="Описание алгоритма и выводы в TOML")
    report.add_argument("--benchmarks", type=Path, help="CSV с реальными измерениями алгоритма")
    return p


def result_status(result):
    print(
        f"Статус: {result['status']}; запросов: {result['metrics']['requests']}; "
        f"токенов (известно): {result['metrics']['total_tokens_known']}; "
        f"время: {result['metrics']['active_time_s']:.2f} с"
    )
    if result.get("error"):
        print(result["error"], file=sys.stderr)
    return 0 if result["status"] in {"passed", "unverified"} else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        overrides = {"model": {"name": args.model}} if args.model else {}
        if args.command == "experiment" and args.repeats is not None:
            overrides["experiment"] = {"repeats": args.repeats}
        config = load_config(args.config, overrides)
        if args.command == "eco-setup":
            install = setup_eco(args.source_dir)
            if not args.skip_smoke:
                smoke_test(install)
            return 0
        if args.command == "eco-doctor":
            smoke_test(installed())
            return 0
        if args.command == "lab-init":
            init_lab(args.directory, args.topic, args.name, example=args.example)
            return 0
        if args.command == "lab-run":
            _, _, passed = run_lab(
                args.directory,
                config_path=args.config,
                model=args.model,
                repeats=args.repeats,
                auto_server=not args.no_server,
            )
            return 0 if passed else 1
        if args.command == "lab-resume":
            if args.config or args.model:
                raise ValueError("lab-resume uses the configuration saved in run.json")
            prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
            result = resume_lab(
                args.directory,
                args.run,
                prompt,
                add_requests=args.add_requests,
                add_tokens=args.add_tokens,
                add_seconds=args.add_seconds,
                add_context_chars=args.add_context_chars,
                compact=args.compact_history,
                auto_server=not args.no_server,
            )
            return result_status(result)
        if args.command == "lab-report":
            directory = args.directory.resolve()
            manifest = read_json(directory / "lab.json")
            if not args.experiment and not manifest["experiments"]:
                raise ValueError("No experiment has been run")
            path = args.experiment.resolve() if args.experiment else directory / manifest["experiments"][-1]
            target = directory / "reports" / path.name
            make_lab_reports(directory, load_runs([path]), target)
            print("Отчёт: " + str(target / "comparison.docx"))
            return 0
        if args.command == "doctor":
            info = environment_info()
            info["mlx_platform"] = sys.platform == "darwin" and platform.machine() == "arm64"
            info["clang"] = shutil.which("clang")
            info["sandbox_exec"] = sandbox_available()
            info["base_url"] = config["model"]["base_url"]
            try:
                info["server"] = Client(config).models()
                ok = True
            except Exception as exc:
                info["server"] = "Недоступен: " + str(exc)
                ok = False
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0 if ok else 1
        if args.command == "serve":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                raise ValueError("MLX profile requires macOS Apple Silicon; use another API endpoint")
            extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
            # No cache reuse across experiments; warmup loads GPU/model but does not share task context.
            command = [
                sys.executable,
                "-m",
                "mlx_lm",
                "server",
                "--model",
                config["model"]["name"],
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--prompt-cache-size",
                "0",
                "--prefill-step-size",
                "512",
                *extra,
            ]
            env = os.environ.copy()
            env.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
            # mlx-lm scans the HF cache for /models even when weights are local.
            cache = Path(env.get("HF_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub")))
            cache.mkdir(parents=True, exist_ok=True)
            metadata = Path(".cache/arpo-server.json")
            metadata.parent.mkdir(parents=True, exist_ok=True)
            save_json(
                metadata,
                {
                    "started_at": utc_now(),
                    "command": command,
                    "model": config["model"]["name"],
                    "port": args.port,
                    "environment": environment_info(),
                },
            )
            print("Сервер:", " ".join(command), flush=True)
            return subprocess.call(command, env=env)
        if args.command == "download":
            from huggingface_hub import HfApi, snapshot_download

            model = config["model"]["name"]
            target = args.destination.resolve()
            record = target / "arpo-model.json"
            if record.is_file() and read_json(record)["repo_id"] != model:
                raise ValueError("Destination contains a different model; choose another --destination")
            sha = HfApi().model_info(model, revision=args.revision).sha
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                model,
                revision=sha,
                local_dir=target,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "README.md"],
            )
            save_json(target / "arpo-model.json", {"repo_id": model, "revision": sha})
            print(f"Модель: {target}\nРевизия: {sha}\nДля запуска: arpo --model '{target}' serve")
            return 0
        if args.command == "warmup":
            warm = deepcopy(config)
            warm["model"]["max_tokens"] = 8
            content = Client(warm).complete(
                [{"role": "user", "content": "Reply OK."}],
                lambda event: print(json.dumps({"usage": event["usage"], "duration_s": event["duration_s"]})),
            )
            print(content)
            return 0
        if args.command == "run":
            prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError("Prompt must not be empty")
            path = new_run(config, prompt, args.output, label=args.label, seed=args.seed_dir)
            print(f"Запуск: {path}")
            return result_status(run_session(path))
        if args.command == "resume":
            path = args.run.resolve()
            data = read_json(path / "run.json")
            if args.config or args.model:
                raise ValueError("resume uses the saved configuration; --config/--model cannot change it")
            followup = (
                args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
            )
            if not followup.strip():
                raise ValueError("Followup must not be empty")
            increase_run_budgets(data, requests=args.add_requests, tokens=args.add_tokens,
                                 seconds=args.add_seconds, context_chars=args.add_context_chars)
            if args.compact_history:
                compact_history(path, data)
            save_json(path / "run.json", data)
            return result_status(run_session(path, followup))
        if args.command == "recover":
            path = args.run.resolve()
            data = read_json(path / "run.json")
            if data["status"] != "running":
                raise ValueError("Run is not marked running")
            data["status"] = "interrupted"
            data["metrics"]["timing_complete"] = False
            data["metrics"]["requests_complete"] = False
            data["metrics"]["tokens_complete"] = False
            data["recovery_note"] = "Recovered after abrupt termination; unfinished session time is unknown"
            save_json(path / "run.json", data)
            print("Статус восстановлен. Метрики незавершённой сессии могут быть неполными.")
            return 0
        if args.command == "experiment":
            path = experiment(config, args.short, args.detailed, args.output, args.seed_dir)
            print(f"Эксперимент: {path}")
            results = load_runs([path])
            return 0 if results and all(r["status"] == "passed" for r in results) else 1
        if args.command == "report":
            for path in write_report(
                load_runs(args.runs), args.output, args.formats.split(","), args.notes, args.benchmarks
            ):
                print(path)
            return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0
