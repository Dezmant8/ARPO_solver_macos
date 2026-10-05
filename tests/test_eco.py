import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from arpo_lab.agent import new_run, validate
from arpo_lab.config import load_config
from arpo_lab.eco import extract_selected, installed, setup_eco
from arpo_lab.lab import prepare_abs
from arpo_lab.local_server import local_server
from arpo_lab.protocol import files_reply, native_reply, qwen_xml_reply
from arpo_lab.storage import read_json
from arpo_lab.workspace import Workspace, execute, sandbox_available


class EcoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_rejects_traversal_and_symlinks_even_in_unselected_entries(self):
        for name, mode in [("../outside", stat.S_IFREG), ("Eco.Framework/link", stat.S_IFLNK)]:
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                info = zipfile.ZipInfo(name)
                info.external_attr = (mode | 0o777) << 16
                stream.writestr(info, "outside")
            with self.subTest(name=name), self.assertRaises(ValueError):
                extract_selected(archive, self.root / "out", lambda _: False)
        self.assertFalse((self.root.parent / "outside").exists())

    def test_setup_is_repeatable_and_detects_changed_sdk(self):
        source = self.root / "Design operating systems"
        source.mkdir()
        with zipfile.ZipFile(source / "Eco.Framework_20260918.zip", "w") as stream:
            stream.writestr("Eco.Framework/Eco.Core1/SharedFiles/IEcoBase1.h", "base")
            stream.writestr("Eco.Framework/Eco.Core1/BuildFiles/Windows/file.dll", "windows")
        with zipfile.ZipFile(source / "eco-wizard-macos-arm64-0.2.2.zip", "w") as stream:
            stream.writestr("eco-wizard/eco-wizard", "binary")
        version = type("Result", (), {"stdout": json.dumps({"version": "0.2.2"})})()
        with (
            patch("arpo_lab.eco.require_apple_silicon"),
            patch("arpo_lab.eco.subprocess.run", return_value=version),
        ):
            first = setup_eco(source, self.root, progress=lambda _: None)
            second = setup_eco(source, self.root, progress=lambda _: None)
        self.assertEqual(first["sdk_sha256"], second["sdk_sha256"])
        self.assertFalse(list(Path(first["sdk"]).rglob("*.dll")))
        self.assertEqual(installed(self.root)["wizard_version"], "0.2.2")
        (Path(first["sdk"]) / "Eco.Core1/SharedFiles/IEcoBase1.h").write_text("changed")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            installed(self.root)

    def test_read_only_paths_reject_file_tools(self):
        sdk = self.root / ".sdk"
        sdk.mkdir()
        config = load_config(overrides={"tools": {"read_only_paths": [".sdk"], "sandbox": "off"}})
        tools = Workspace(self.root, config)
        with self.assertRaisesRegex(ValueError, "read-only"):
            tools.action({"type": "write", "path": ".sdk/header.h", "content": "modified"}, 10)
        tools.action({"type": "write", "path": "source.c", "content": "allowed"}, 10)
        for path in ["/tmp", "../tests", "."]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_config(overrides={"tools": {"read_only_paths": [path]}})

    @unittest.skipUnless(sandbox_available(), "macOS sandbox required")
    def test_subprocess_cannot_modify_sdk(self):
        sdk = self.root / ".sdk"
        sdk.mkdir()
        target = sdk / "header.h"
        target.write_text("original")
        config = load_config(overrides={"tools": {"sandbox": "required", "read_only_paths": [".sdk"]}})
        result = execute(
            [sys.executable, "-c", "from pathlib import Path; Path('.sdk/header.h').write_text('bad')"],
            self.root,
            config,
        )
        self.assertNotEqual(result["exit_code"], 0)
        self.assertEqual(target.read_text(), "original")

    def test_validator_change_between_iterations_is_not_accepted(self):
        script = self.root / "check.py"
        script.write_text("raise SystemExit(0)")
        config = load_config(
            overrides={
                "tools": {"sandbox": "off"},
                "validation": {"commands": [[sys.executable, str(script)]]},
            }
        )
        run = new_run(config, "task", self.root / "runs")
        script.write_text("# a different check\nraise SystemExit(0)")
        data = read_json(run / "run.json")
        result = validate(run, data, lambda: 10)
        self.assertFalse(result["passed"])
        self.assertIn("changed", result["checks"][0]["output"])

    def test_reuses_existing_server_without_terminating_it(self):
        config = load_config()
        with (
            patch("arpo_lab.local_server.check_model"),
            patch("arpo_lab.local_server.subprocess.Popen") as launch,
        ):
            with local_server(config, self.root, progress=lambda _: None):
                pass
        launch.assert_not_called()

    def test_empty_native_arguments_are_not_silently_treated_as_root_listing(self):
        message = {"tool_calls": [{"id": "missing", "function": {"name": "list_files", "arguments": "{}"}}]}
        action = native_reply(message, 12)["actions"][0]
        self.assertEqual(action["type"], "invalid")
        self.assertIn("path", action["error"])
        with self.assertRaisesRegex(ValueError, "path"):
            Workspace(self.root, load_config()).action(action, 1)

    def test_qwen_xml_preserves_literal_code_and_handles_missing_outer_tag(self):
        source = 'if (x < 0 && y > 1) return "a&b";\n'
        raw = "<function=write_file>\n<parameter=path>\nSourceFiles/test.c\n</parameter>\n"
        raw += "<parameter=content>\n" + source + "</parameter>\n</function>\n</tool_call><|im_end|>"
        action = native_reply(qwen_xml_reply(raw), 12)["actions"][0]
        self.assertEqual(action["path"], "SourceFiles/test.c")
        self.assertEqual(action["content"], source)
        run = "<tool_call><function=run_command><parameter=argv>['clang', '-std=c89']"
        run += "</parameter></function></tool_call>"
        self.assertEqual(native_reply(qwen_xml_reply(run), 12)["actions"][0]["argv"], ["clang", "-std=c89"])

    def test_initial_context_is_identical_and_excludes_assignment_pdf(self):
        seed = self.root / "seed"
        (seed / "SharedFiles").mkdir(parents=True)
        (seed / "SharedFiles/api.h").write_text("int unsolved(int);\n")
        (seed / "assignment.pdf").write_text("private task list")
        config = load_config(
            overrides={"agent": {"workspace_manifest": True, "context_files": ["SharedFiles/*.h"]}}
        )
        short = new_run(config, "short", self.root / "runs", seed=seed)
        detailed = new_run(config, "detailed", self.root / "runs", seed=seed)
        left, right = read_json(short / "run.json"), read_json(detailed / "run.json")
        self.assertEqual(left["messages"][0], right["messages"][0])
        self.assertIn("int unsolved(int)", left["messages"][0]["content"])
        self.assertNotIn("assignment.pdf", left["messages"][0]["content"])
        self.assertGreater(left["seed_context_chars"], 0)

    def test_multiple_literal_files_are_submitted_together_and_alternatives_rejected(self):
        reply = files_reply(
            "```file:SourceFiles/lab.c\nint answer(void) { return 42; }\n```\n"
            '```file:UnitTestFiles/SourceFiles/main.c\nconst char* label = "a\\\\b";\n```',
            12,
        )
        self.assertTrue(reply["done"])
        self.assertEqual(len(reply["actions"]), 2)
        self.assertIn('"a\\\\b"', reply["actions"][1]["content"])
        for text in [
            "Here is a solution, but no files",
            "```file:test.c\nincomplete",
            "```file:test.c\na\n```\n```file:test.c\nb\n```",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                files_reply(text, 12)

    def test_abs_contract_and_reference_factory_are_materialized_without_solution(self):
        project = self.root / "seed"
        (project / "SharedFiles").mkdir(parents=True)
        (project / "UnitTestFiles/SourceFiles").mkdir(parents=True)
        (project / "SourceFiles").mkdir()
        checks = self.root / "checks"
        checks.mkdir()
        (project / "SharedFiles/IEcoTest.h").write_text(
            "    int16_t (ECOCALLMETHOD *MyFunction)(IEcoTest* me, char_t* Name, char_t** CopyName);"
        )
        (project / "SharedFiles/IdEcoTest.h").write_text("GetIEcoComponentFactoryPtr_" + "A" * 32)
        (project / "SourceFiles/CEcoTest.c").write_text(
            "static int16_t ECOCALLMETHOD CEcoTest_ABC_MyFunction(IEcoTestPtr_t me) {\n"
            "    return 0;\n}\n"
            "IEcoTestVTbl table = { CEcoTest_ABC_MyFunction };\n"
        )
        prepare_abs(project, checks, "EcoTest")
        header = (project / "SharedFiles/IEcoTest.h").read_text()
        self.assertIn("AbsLongDouble", header)
        self.assertIn("IEcoTestPtr_t me", header)
        self.assertNotIn("MyFunction", header)
        reference = (checks / "reference.c").read_text()
        self.assertIn("GetIEcoComponentFactoryPtr_" + "A" * 32, reference)
        self.assertNotIn("Ptr_GUID", reference)
        self.assertIn("QueryComponent", (project / "UnitTestFiles/SourceFiles/EcoTest.c").read_text())
        methods = (project / "SourceFiles/AbsMethods.c").read_text()
        self.assertNotIn("return", methods)
        self.assertIn("No solution", methods)
        self.assertIn("lab_abs_int", (project / "SourceFiles/CEcoTest.c").read_text())


if __name__ == "__main__":
    unittest.main()
