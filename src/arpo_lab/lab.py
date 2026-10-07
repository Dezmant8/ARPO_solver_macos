"""Prepare a versioned ACOM experiment and produce one comparison report."""

import json
import re
import sys
import tempfile
from pathlib import Path

from .agent import compact_history, run_session
from .config import increase_run_budgets, load_config
from .eco import generate_project, installed
from .eco_build import build
from .experiment import experiment
from .local_server import local_server
from .report import load_runs, write_report
from .storage import file_hash, read_json, save_json, utc_now


def quoted(text):
    return json.dumps(str(text), ensure_ascii=False)


def init_lab(directory, topic, name="Eco.Lab1", repo=Path("."), example=None):
    repo, directory = Path(repo).resolve(), Path(directory).resolve()
    if directory.exists():
        raise ValueError("Lab directory already exists; choose a new name to preserve experiments")
    install = installed(repo)
    directory.mkdir(parents=True)
    project = generate_project(directory / "seed", name, install)
    checks = directory / "checks"
    checks.mkdir()
    builder = Path(__file__).with_name("eco_build.py").resolve()
    command = [
        str(sys.executable),
        str(builder),
        "--project",
        "{workspace}",
        "--run",
        "--sdk-sha256",
        install["sdk_sha256"],
    ]
    validations = [command]
    if example == "abs":
        stem = read_json(project / "arpo-scaffold.json")["receipt"]["normalized_name"]
        prepare_abs(project, checks, stem)
        with tempfile.TemporaryDirectory(prefix="abi-check-", dir=project) as probe:
            build(project, compile_only=True, output=Path(probe).relative_to(project).as_posix())
        validations.append(
            [
                str(sys.executable),
                str(builder),
                "--project",
                "{workspace}",
                "--client",
                str(checks / "reference.c"),
                "--run",
                "--sanitize",
                "--sdk-sha256",
                install["sdk_sha256"],
            ]
        )
        short = "Implement absolute value for all functions declared in ArpoMethods.h. "
        short += "Use ANSI C89 and no libc calls. Return one full source file including ArpoMethods.h."
        detailed = short + "\nCopy the function declarations from ArpoMethods.h exactly. "
        detailed += "AbsInt and AbsLong return unsigned magnitudes, including minimum signed inputs. "
        detailed += "Convert to unsigned first, then subtract from unsigned zero for negative inputs; "
        detailed += "never negate the signed minimum. "
        detailed += "AbsFloat, AbsDouble and AbsLongDouble return the magnitude without losing precision. "
        detailed += "Canonicalize either signed zero to positive zero using x == 0 before the sign branch. "
        detailed += "Every method returns 0 on success and -1 if me == 0 or out == 0. "
        detailed += "Do not allocate memory or call math.h, libc or third party libraries. "
        detailed += (
            "The ACOM factory, lifetime methods and console EcoMain client are already supplied. "
            "Do not redeclare interfaces, IID/CID or factory methods. Implement only the five "
            "functions declared in ArpoMethods.h, with exactly their names and SDK types."
        )
        protected = [
            ".sdk",
            ".arpo-build.py",
            "arpo-scaffold.json",
            ".arpo-toolchain.json",
            "SharedFiles",
            "HeaderFiles",
            "UnitTestFiles",
            f"SourceFiles/C{stem}.c",
            f"SourceFiles/C{stem}Factory.c",
        ]
    else:
        check = checks / "validate.py"
        check.write_text(
            '"""Replace with independent task-specific checks before running.\n'
            "Run from the model workspace; SDK and C89 build are already checked separately.\n"
            'Do not accept an answer merely because its own client exits 0.\n"""\n'
            'raise SystemExit("ARPO_CHECKS_TODO: implement independent numerical/algorithm tests")\n',
            encoding="utf-8",
        )
        validations.append([str(sys.executable), str(check)])
        short = f"Выполни лабораторную ACOM на тему «{topic}» на C89 в имеющемся проекте."
        detailed = short + "\n[Уточнить интерфейс, алгоритмы, типы, граничные случаи и критерии принятия.]"
        protected = [".sdk", ".arpo-build.py", "arpo-scaffold.json", ".arpo-toolchain.json"]
    (directory / "short.txt").write_text(short + "\n", encoding="utf-8")
    (directory / "detailed.txt").write_text(detailed + "\n", encoding="utf-8")
    model = repo / "models/qwen-coder"
    model_info = repo / ".eco/model.json"
    if model_info.exists():
        model = Path(read_json(model_info)["path"])
    text = f"[model]\nname = {quoted(model)}\nmax_tokens = 8192\ntimeout_s = 600\n\n"
    text += '[agent]\nprotocol = "files"\nmax_requests = 30\nmax_total_tokens = 600000\n'
    text += "workspace_manifest = true\n"
    text += 'context_files = ["SharedFiles/*.h", "HeaderFiles/*.h", "SourceFiles/*.c", '
    text += '"UnitTestFiles/SourceFiles/*.c", ".sdk/Eco.System1/SharedFiles/*.h", '
    text += '".sdk/Eco.InterfaceBus1/SharedFiles/*.h", ".sdk/Eco.MemoryManager1/SharedFiles/*.h"]\n'
    text += "max_wall_s = 3600\nmax_context_chars = 220000\n\n[tools]\n"
    text += 'allowed_executables = ["clang", "cc", "ar", "nm", "make", "python3", "ls", "find", "cat"]\n'
    text += 'sandbox = "required"\ntimeout_s = 120\n'
    text += "read_only_paths = " + json.dumps(protected) + "\n\n[validation]\n"
    text += "timeout_s = 180\ncommands = " + json.dumps(validations, ensure_ascii=False) + "\n\n"
    text += '[experiment]\nrepeats = 3\norder = "alternate"\nwarmup = true\n\n[report]\n'
    text += f'title = {quoted("Лабораторная: " + topic)}\nauthor = ""\ngroup = ""\n'
    if example == "abs":
        text = text.replace("max_requests = 30", "max_requests = 12")
        text = text.replace(
            'protocol = "files"', 'protocol = "single-file"\nfallback_file = "SourceFiles/AbsMethods.c"'
        )
        text = text.replace("max_tokens = 8192", "max_tokens = 4096")
        text = text.replace("workspace_manifest = true", "workspace_manifest = false")
        text = re.sub(
            r"context_files = .*\n",
            'context_files = ["HeaderFiles/ArpoMethods.h"]\n',
            text,
        )
        text = text.replace(
            "timeout_s = 600\n", "timeout_s = 600\n\n[model.extra_body]\nrepetition_penalty = 1.0\n"
        )
    (directory / "config.toml").write_text(text, encoding="utf-8")
    notes = {
        "task": topic,
        "integration": "Eco SDK "
        + install["framework_archive"]
        + "; SHA256: "
        + install["framework_archive_sha256"]
        + ". C89 component, static factory registration, EcoMain console client.",
        "algorithm": "[Заполнить по исходникам обеих проверенных реализаций.]",
        "complexity": "[Обосновать время и память по фактическому алгоритму.]",
        "improvements": "[Описать реализованные изменения или указать, что их нет.]",
        "testing": "[Описать независимые проверки, диапазоны и измеренную точность.]",
        "visualization": "[Добавить реальные графики, если нужны по заданию.]",
        "conclusion": "[Сравнить метрики, успешность и ограничения после эксперимента.]",
    }
    notes_text = "\n".join(f"{key} = {quoted(value)}" for key, value in notes.items()) + "\n"
    (directory / "notes-comparison.toml").write_text(notes_text, encoding="utf-8")
    save_json(
        directory / "lab.json",
        {
            "schema_version": 1,
            "topic": topic,
            "name": name,
            "created_at": utc_now(),
            "seed": str(project.relative_to(directory)),
            "sdk": install,
            "example": example,
            "experiments": [],
        },
    )
    description = (
        "Пять методов фиксированного интерфейса: AbsInt (int32 → uint32), AbsLong (int64 → uint64), "
        "AbsFloat, AbsDouble, AbsLongDouble. Результат передаётся через out, статус 0 означает успех. "
        "Для целых проверяются ноль, положительные, отрицательные и минимальные знаковые значения; "
        "для вещественных — крайние конечные значения, малые числа и нормализация signed zero. "
        "Null me/out должны приводить к ненулевому статусу. QueryInterface, AddRef/Release и фабрика "
        "проверяются через настоящий Eco bus. Компонент использует C89 и не вызывает libc. "
        "Эталонный клиент использует fabs; 10 случаев double выводятся в CSV, миллион вызовов "
        "измеряется CPU таймером clock. NaN и бесконечности не входят в заданную область проверки. "
        "Общая фабрика, управление ссылками, интерфейс и консольный клиент поставлены в seed. "
        "Qwen генерирует только пять численных методов в AbsMethods.c. Стоимость этого эксперимента "
        "не измеряет создание всего ACOM каркаса. Изначальный шаблон не содержит численного "
        "решения и не проходит независимые проверки."
        if example == "abs"
        else "Допишите контракт и критерии перед экспериментом."
    )
    (directory / "task.md").write_text(
        f"# {topic}\n\n{description}\n\nПроверки лежат в checks и не передаются локальной модели. "
        "Оба промпта получают одинаковый seed. PDF заданий модели не передаются.\n",
        encoding="utf-8",
    )
    print("Лабораторная подготовлена: " + str(directory))
    return directory


def run_lab(directory, repo=Path("."), config_path=None, model=None, repeats=None, auto_server=True):
    directory, repo = Path(directory).resolve(), Path(repo).resolve()
    manifest = read_json(directory / "lab.json")
    config = load_config(
        config_path or directory / "config.toml", {"model": {"name": model}} if model else None
    )
    if repeats is not None:
        config["experiment"]["repeats"] = repeats
        config = load_config(overrides=config)
    if len(config["validation"]["commands"]) < 2:
        raise ValueError("Need both an Eco build check and independent task-specific checks")
    if not manifest.get("example"):
        check = directory / "checks/validate.py"
        if check.exists() and "ARPO_CHECKS_TODO" in check.read_text():
            raise ValueError("Independent task tests are unfinished: " + str(check))
    for label in ["short", "detailed"]:
        text = (directory / f"{label}.txt").read_text(encoding="utf-8")
        if not text.strip() or "[Уточнить" in text:
            raise ValueError("Finish the prompt first: " + label + ".txt")
    # Record every externally supplied validator before any generation begins.
    check_files = {
        str(Path(arg).resolve()): file_hash(arg)
        for command in config["validation"]["commands"]
        for arg in command[1:]
        if "{workspace}" not in arg and Path(arg).is_file()
    }
    with local_server(config, repo, enabled=auto_server):
        path = experiment(
            config,
            directory / "short.txt",
            directory / "detailed.txt",
            directory / "runs",
            directory / manifest["seed"],
        )
    manifest["experiments"].append(str(path.relative_to(directory)))
    save_json(directory / "lab.json", manifest)
    save_json(
        path / "lab-inputs.json",
        {
            "topic": manifest["topic"],
            "sdk": manifest["sdk"],
            "validators": check_files,
            "prepared_at": utc_now(),
        },
    )
    runs = load_runs([path])
    target = directory / "reports" / path.name
    make_lab_reports(directory, runs, target)
    print("Результаты: " + str(path) + "\nОтчёт: " + str(target / "comparison.docx"))
    return path, target, bool(runs) and all(run["status"] == "passed" for run in runs)


def make_lab_reports(directory, runs, target):
    directory, target = Path(directory), Path(target)
    formats = ("md", "html", "csv", "json", "docx")
    csv_path = directory / "benchmarks-comparison.csv"
    write_report(
        runs, target, formats, directory / "notes-comparison.toml", csv_path if csv_path.is_file() else None
    )


def resume_lab(
    directory, run, prompt, *, add_requests=0, add_tokens=0, add_seconds=0,
    add_context_chars=0, compact=False, auto_server=True
):
    directory, run = Path(directory).resolve(), Path(run).resolve()
    if not run.is_relative_to(directory / "runs"):
        raise ValueError("Run must belong to this laboratory's runs directory")
    if not prompt.strip():
        raise ValueError("Nonempty followup and nonnegative budget increments are required")
    data = read_json(run / "run.json")
    if data["status"] == "running":
        raise ValueError("Previous process is marked running; stop it and use recover first")
    increase_run_budgets(data, requests=add_requests, tokens=add_tokens,
                         seconds=add_seconds, context_chars=add_context_chars)
    with local_server(data["config"], Path.cwd(), enabled=auto_server):
        if compact:
            compact_history(run, data)
        save_json(run / "run.json", data)
        result = run_session(run, prompt)
    target = directory / "reports" / run.parent.name
    make_lab_reports(directory, load_runs([run.parent]), target)
    print("Отчёт обновлён: " + str(target / "comparison.docx"))
    return result


def prepare_abs(project, checks, stem):
    """An unsolved ACOM lab with a fixed API and independent trusted reference client."""
    header = project / "SharedFiles" / f"I{stem}.h"
    text = header.read_text(encoding="utf-8-sig")
    methods = """    int16_t (ECOCALLMETHOD *AbsInt)(I@Ptr_t me, int32_t x, uint32_t* out);
    int16_t (ECOCALLMETHOD *AbsLong)(I@Ptr_t me, int64_t x, uint64_t* out);
    int16_t (ECOCALLMETHOD *AbsFloat)(I@Ptr_t me, float x, float* out);
    int16_t (ECOCALLMETHOD *AbsDouble)(I@Ptr_t me, double x, double* out);
    int16_t (ECOCALLMETHOD *AbsLongDouble)(I@Ptr_t me, long double x, long double* out);""".replace("@", stem)
    text, count = re.subn(r"    int16_t \(ECOCALLMETHOD \*MyFunction\)[^;]+;", methods, text)
    if count != 1:
        raise ValueError("Wizard interface format changed; cannot prepare abs example safely")
    header.write_text(text, encoding="utf-8")
    (project / "UnitTestFiles/SourceFiles" / f"{stem}.c").write_text(
        '#include "IEcoSystem1.h"\nint16_t EcoMain(IEcoUnknown* system) {\n'
        "    (void)system;\n    return -1; /* implement a real console client */\n}\n",
        encoding="utf-8",
    )
    # No numerical implementation is supplied. Common factory/lifetime/client wiring is seed data.
    identifier = (project / "SharedFiles" / f"Id{stem}.h").read_text(encoding="utf-8-sig")
    match = re.search(r"GetIEcoComponentFactoryPtr_([A-Fa-f0-9]{32})", identifier)
    if not match:
        raise ValueError("Cannot find the generated static component factory")
    reference = ABS_REFERENCE.replace("@", stem).replace("Ptr_GUID", "Ptr_" + match[1])
    (checks / "reference.c").write_text(reference, encoding="utf-8")
    wire_abs_methods(project, stem, match[1])


def wire_abs_methods(project, stem, guid):
    """Connect unsolved method symbols to the vendor vtable; supply no numerical algorithm."""
    header = project / "HeaderFiles/ArpoMethods.h"
    header.parent.mkdir(exist_ok=True)
    prototypes = (
        "/* Fixed ABI; implementations belong in SourceFiles/AbsMethods.c. */\n"
        '#ifndef ARPO_METHODS_H\n#define ARPO_METHODS_H\n#include "I@.h"\n'
        "int16_t ECOCALLMETHOD lab_abs_int(I@* me, int32_t x, uint32_t* out);\n"
        "int16_t ECOCALLMETHOD lab_abs_long(I@* me, int64_t x, uint64_t* out);\n"
        "int16_t ECOCALLMETHOD lab_abs_float(I@* me, float x, float* out);\n"
        "int16_t ECOCALLMETHOD lab_abs_double(I@* me, double x, double* out);\n"
        "int16_t ECOCALLMETHOD lab_abs_long_double(I@* me, long double x, long double* out);\n"
        "#endif\n"
    ).replace("@", stem)
    header.write_text(prototypes, encoding="utf-8")
    source = project / "SourceFiles" / f"C{stem}.c"
    text = source.read_text(encoding="utf-8-sig")
    method = re.search(r"static int16_t ECOCALLMETHOD (\w+_MyFunction)\([^\n]+\) \{.*?^\}", text, re.M | re.S)
    if not method:
        raise ValueError("Cannot locate vendor MyFunction body for the ACOM method adapter")
    name = method[1]
    text = text[: method.start()] + text[method.end() :]
    replacement = ",\n    ".join(
        "lab_abs_" + kind for kind in ["int", "long", "float", "double", "long_double"]
    )
    if text.count(name) != 1:
        raise ValueError("Expected one original method slot in vendor vtable")
    text = '#include "ArpoMethods.h"\n' + text.replace(name, replacement)
    source.write_text(text, encoding="utf-8")
    (project / "SourceFiles/AbsMethods.c").write_text(
        '#include "ArpoMethods.h"\n/* Implement all five declared functions. No solution is supplied. */\n',
        encoding="utf-8",
    )
    (project / "UnitTestFiles/SourceFiles" / f"{stem}.c").write_text(
        ABS_CONSOLE.replace("@", stem).replace("Ptr_GUID", "Ptr_" + guid), encoding="utf-8"
    )


ABS_CONSOLE = r"""/* Common console client, identical in both experimental branches. */
#include <stdio.h>
#include "IEcoSystem1.h"
#include "IdEcoInterfaceBus1.h"
#include "Id@.h"
int16_t EcoMain(IEcoUnknown* system) {
    IEcoSystem1* sys = 0;
    IEcoInterfaceBus1* bus = 0;
    I@* lab = 0;
    uint32_t ui = 0;
    uint64_t ul = 0;
    float f = 0;
    double d = 0;
    long double ld = 0;
    int16_t status = 0;
    if (!system || system->pVTbl->QueryInterface(system, &GID_IEcoSystem, (void**)&sys) != 0 || !sys)
        return 1;
    if (sys->pVTbl->QueryInterface(sys, &IID_IEcoInterfaceBus1, (void**)&bus) != 0 || !bus) {
        sys->pVTbl->Release(sys); return 2;
    }
    if (bus->pVTbl->RegisterComponent(bus, &CID_@, (IEcoUnknown*)GetIEcoComponentFactoryPtr_GUID) != 0
        || bus->pVTbl->QueryComponent(bus, &CID_@, 0, &IID_I@, (void**)&lab) != 0 || !lab) {
        bus->pVTbl->Release(bus); sys->pVTbl->Release(sys); return 3;
    }
    if (lab->pVTbl->AbsInt(lab, -42, &ui) != 0 || lab->pVTbl->AbsLong(lab, -42, &ul) != 0
        || lab->pVTbl->AbsFloat(lab, -1.25f, &f) != 0 || lab->pVTbl->AbsDouble(lab, -1.25, &d) != 0
        || lab->pVTbl->AbsLongDouble(lab, -1.25L, &ld) != 0) status = 4;
    printf("int=%u long=%lu float=%g double=%g long_double=%Lg\n", ui, (unsigned long)ul, (double)f, d, ld);
    lab->pVTbl->Release(lab);
    bus->pVTbl->Release(bus);
    sys->pVTbl->Release(sys);
    return status;
}
"""


ABS_REFERENCE = r"""/* Trusted reference client; permitted libc is confined to this test. */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "IEcoSystem1.h"
#include "IdEcoInterfaceBus1.h"
#include "Id@.h"

int16_t EcoMain(IEcoUnknown* system) {
    IEcoSystem1* sys = 0;
    IEcoInterfaceBus1* bus = 0;
    I@* lab = 0;
    IEcoUnknown* unknown = 0;
    UGUID missing = {0x01, 0x10, {0}};
    double data[] = {-0.0, 0.0, -1.0, 1.0, -1.25, 1.25, -1e-200, 1e-200, -1e200, 1e200};
    double value = 0.0;
    float f = 0.0f;
    long double ld = 0.0L;
    uint32_t ui = 0;
    uint64_t ul = 0;
    int64_t half = (int64_t)1 << 62;
    uint32_t refs = 0;
    int i = 0;
    int failed = 0;
    clock_t started = 0;
    double seconds = 0.0;
    if (!system || system->pVTbl->QueryInterface(system, &GID_IEcoSystem, (void**)&sys) != 0 || !sys)
        return 1;
    if (sys->pVTbl->QueryInterface(sys, &IID_IEcoInterfaceBus1, (void**)&bus) != 0 || !bus) {
        sys->pVTbl->Release(sys); return 2;
    }
    if (bus->pVTbl->RegisterComponent(bus, &CID_@, (IEcoUnknown*)GetIEcoComponentFactoryPtr_GUID) != 0
        || bus->pVTbl->QueryComponent(bus, &CID_@, 0, &IID_I@, (void**)&lab) != 0 || !lab) {
        bus->pVTbl->Release(bus); sys->pVTbl->Release(sys); return 3;
    }
    if (lab->pVTbl->QueryInterface(lab, &IID_IEcoUnknown, (void**)&unknown) != 0 || !unknown) failed++;
    if (unknown) unknown->pVTbl->Release(unknown);
    unknown = 0;
    if (lab->pVTbl->QueryInterface(lab, &missing, (void**)&unknown) == 0 || unknown) failed++;
    if (lab->pVTbl->QueryInterface(lab, &IID_IEcoUnknown, 0) == 0) failed++;
    refs = lab->pVTbl->AddRef(lab);
    if (refs < 2 || lab->pVTbl->Release(lab) != refs - 1) failed++;
    if (lab->pVTbl->AbsInt(lab, -2147483647 - 1, &ui) != 0 || ui != 2147483648U) failed++;
    if (lab->pVTbl->AbsInt(lab, -1, &ui) != 0 || ui != 1U) failed++;
    if (lab->pVTbl->AbsInt(lab, 0, &ui) != 0 || ui != 0U) failed++;
    if (lab->pVTbl->AbsInt(lab, 42, &ui) != 0 || ui != 42U) failed++;
    if (lab->pVTbl->AbsLong(lab, -half - half, &ul) != 0 || ul != ((uint64_t)1 << 63)) failed++;
    if (lab->pVTbl->AbsLong(lab, -1, &ul) != 0 || ul != 1) failed++;
    if (lab->pVTbl->AbsLong(lab, 0, &ul) != 0 || ul != 0) failed++;
    if (lab->pVTbl->AbsLong(lab, 42, &ul) != 0 || ul != 42) failed++;
    if (lab->pVTbl->AbsFloat(lab, -FLT_MAX, &f) != 0 || f != FLT_MAX) failed++;
    if (lab->pVTbl->AbsFloat(lab, -FLT_MIN, &f) != 0 || f != FLT_MIN) failed++;
    if (lab->pVTbl->AbsFloat(lab, -0.0f, &f) != 0 || f != 0.0f || 1.0f / f < 0.0f) failed++;
    if (lab->pVTbl->AbsLongDouble(lab, -LDBL_MAX, &ld) != 0 || ld != LDBL_MAX) failed++;
    if (lab->pVTbl->AbsLongDouble(lab, -LDBL_MIN, &ld) != 0 || ld != LDBL_MIN) failed++;
    if (lab->pVTbl->AbsLongDouble(lab, -0.0L, &ld) != 0 || ld != 0.0L || 1.0L / ld < 0.0L) failed++;
    if (lab->pVTbl->AbsInt(0, 1, &ui) == 0 || lab->pVTbl->AbsInt(lab, 1, 0) == 0) failed++;
    if (lab->pVTbl->AbsLong(0, 1, &ul) == 0 || lab->pVTbl->AbsLong(lab, 1, 0) == 0) failed++;
    if (lab->pVTbl->AbsFloat(0, 1, &f) == 0 || lab->pVTbl->AbsFloat(lab, 1, 0) == 0) failed++;
    if (lab->pVTbl->AbsDouble(0, 1, &value) == 0 || lab->pVTbl->AbsDouble(lab, 1, 0) == 0) failed++;
    if (lab->pVTbl->AbsLongDouble(0, 1, &ld) == 0 || lab->pVTbl->AbsLongDouble(lab, 1, 0) == 0) failed++;
    printf("input,actual,reference,absolute_error\n");
    for (i = 0; i < 10; ++i) {
        if (lab->pVTbl->AbsDouble(lab, data[i], &value) != 0 || value != fabs(data[i])
            || (value == 0.0 && 1.0 / value < 0.0)) failed++;
        printf("%.17g,%.17g,%.17g,%.17g\n", data[i], value, fabs(data[i]), fabs(value - fabs(data[i])));
    }
    started = clock();
    for (i = 0; i < 1000000; ++i) {
        if (lab->pVTbl->AbsDouble(lab, data[i % 10], &value) != 0) { failed++; break; }
    }
    seconds = (double)(clock() - started) / CLOCKS_PER_SEC;
    printf("ACOM_ABS_CHECK failures=%d calls=1000000 cpu_time_s=%.9f\n", failed, seconds);
    lab->pVTbl->Release(lab);
    bus->pVTbl->Release(bus);
    sys->pVTbl->Release(sys);
    return failed ? 4 : 0;
}
"""
