import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

from arpo_lab.acom import prepare_scaffold, write_contract_check
from arpo_lab.eco import generate_project, installed
from arpo_lab.eco_build import build
from arpo_lab.lab import init_lab, write_review_index

REPO = Path(__file__).resolve().parent.parent


class AcomTests(unittest.TestCase):
    def test_unknown_wizard_abi_is_rejected_without_changing_public_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            (project / "SharedFiles").mkdir()
            public = project / "SharedFiles/IEcoExample.h"
            public.write_text("/* A different SDK contract. */\n")
            (project / "SharedFiles/IdEcoExample.h").write_text(
                "GetIEcoComponentFactoryPtr_" + "A" * 32
            )
            with self.assertRaisesRegex(ValueError, "ABI changed"):
                prepare_scaffold(project, "EcoExample")
            self.assertEqual(public.read_text(), "/* A different SDK contract. */\n")
            self.assertFalse((project / "SourceFiles").exists())

    def test_review_links_use_real_run_sources_and_preserve_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            lab = Path(folder)
            seed = lab / "seed/EcoExample"
            (seed / "SourceFiles").mkdir(parents=True)
            (seed / "SourceFiles/CEcoExample.c").write_text("/* unsolved seed */\n")
            (lab / "checks").mkdir()
            (lab / "lab.json").write_text(json.dumps({"topic": "Example", "seed": "seed/EcoExample"}))
            (lab / "README.md").write_text("# Notes\n\nKeep this paragraph.\n")
            runs = []
            for label, status in [("short", "budget_exhausted"), ("detailed", "passed")]:
                run = lab / "runs" / label
                (run / "workspace/SourceFiles").mkdir(parents=True)
                (run / "workspace/SourceFiles/Methods.c").write_text("/* " + label + " */\n")
                runs.append({"label": label, "status": status, "source_path": str(run), "repeat": 1})
            write_review_index(lab, runs)
            write_review_index(lab, runs)
            text = (lab / "README.md").read_text()
            self.assertIn("Keep this paragraph.", text)
            self.assertEqual(text.count("<!-- arpo:review:start -->"), 1)
            self.assertIn("вариант с кратким промптом | 1 | budget_exhausted", text)
            self.assertIn("вариант с подробным промптом | 1 | passed", text)
            self.assertIn("runs/short/workspace/SourceFiles/Methods.c", text)
            self.assertIn("runs/detailed/workspace/SourceFiles/Methods.c", text)
            for target in re.findall(r"\]\(([^)]+)\)", text):
                self.assertTrue((lab / unquote(target)).exists(), target)
            with tempfile.TemporaryDirectory() as output:
                report = Path(output) / "comparison.docx"
                report.write_bytes(b"report-link fixture")
                write_review_index(lab, runs, report.parent)
                for target in re.findall(r"\]\(([^)]+)\)", (lab / "README.md").read_text()):
                    self.assertTrue((lab / unquote(target)).exists(), target)

    @unittest.skipUnless(sys.platform == "darwin" and (REPO / ".eco/install.json").exists(),
                         "Local Apple toolchain and separately supplied Eco SDK required")
    def test_real_contract_client_detects_ownership_and_aggregation_regressions(self):
        with tempfile.TemporaryDirectory(prefix="acom-regression-") as folder:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                install = installed(REPO)
                project = generate_project(folder, "Eco.ContractRegression", install)
                client = Path(folder) / "acom.c"
                write_contract_check(project, "EcoContractRegression", client)
                source = project / "SourceFiles/CEcoContractRegression.c"
                original = source.read_text()

                def run_contract():
                    diagnostics = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                            result = build(project, install["sdk"], client=client,
                                           output="BuildFiles/Contract", sanitize=True)
                    except RuntimeError as error:
                        raise AssertionError(diagnostics.getvalue()) from error
                    return subprocess.run([result["client"]], capture_output=True, text=True, timeout=20)

                baseline = run_contract()
                self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
                self.assertIn("ACOM_CONTRACT checks=133 failures=0", baseline.stdout)
                mutations = [
                    ("allocator reference leak", "    memory->pVTbl->Release(memory);\n", ""),
                    ("accept unsupported aggregation", "    if (outer) return ERR_ECO_NOAGGREGATION;",
                     "    (void)outer;"),
                ]
                for name, before, after in mutations:
                    with self.subTest(name=name):
                        self.assertIn(before, original)
                        source.write_text(original.replace(before, after, 1))
                        result = run_contract()
                        self.assertNotEqual(result.returncode, 0, "Regression escaped independent checks")
                source.write_text(original)

    @unittest.skipUnless(sys.platform == "darwin" and (REPO / ".eco/install.json").exists(),
                         "Local Apple toolchain and separately supplied Eco SDK required")
    def test_new_abs_lab_has_checked_scaffold_shared_policy_and_unsolved_algorithm(self):
        with tempfile.TemporaryDirectory(prefix="acom-init-") as folder:
            lab = Path(folder) / "absolute"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                init_lab(lab, "Absolute value", "Eco.ContractAbs", REPO, example="abs")
            manifest = json.loads((lab / "lab.json").read_text())
            self.assertEqual(manifest["acom_scaffold_version"], 2)
            self.assertEqual(json.loads((lab / "acom-preflight.json").read_text())["status"], "passed")
            config = (lab / "config.toml").read_text()
            self.assertIn('context_files = ["CODING_STYLE.md", "HeaderFiles/ArpoMethods.h"]', config)
            self.assertIn(str(lab / "checks/acom.c"), config)
            methods = lab / manifest["seed"] / "SourceFiles/AbsMethods.c"
            self.assertNotIn("return", methods.read_text())
            result = subprocess.run(
                [sys.executable, str(REPO / "src/arpo_lab/eco_build.py"),
                 "--project", str(methods.parent.parent), "--run"], capture_output=True, text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, "The unsolved numerical template must not pass")


if __name__ == "__main__":
    unittest.main()
