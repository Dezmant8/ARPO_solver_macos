import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from arpo_lab.agent import new_run, parse_reply, run_session
from arpo_lab.config import load_config
from arpo_lab.protocol import native_reply, single_file_reply
from arpo_lab.report import load_runs, summaries, write_report
from arpo_lab.storage import read_json, tree_hash
from arpo_lab.workspace import Workspace, confined, execute


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, messages, record, can_request, remaining_s):
        content = next(self.replies)
        record(
            {
                "ok": True,
                "duration_s": 0.1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "request": {"messages": deepcopy(messages)},
                "response": {"content": content},
            }
        )
        return content


def reply(text="good"):
    return json.dumps(
        {
            "actions": [{"type": "write", "path": "solution.txt", "content": text}],
            "done": True,
            "summary": "completed",
        }
    )


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.config = load_config(overrides={"tools": {"sandbox": "off"}})

    def tearDown(self):
        self.tmp.cleanup()

    def validator(self):
        script = self.path / "validator.py"
        script.write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.exit(0 if Path('solution.txt').read_text() == 'good' else 1)"
        )
        self.config["validation"]["commands"] = [[sys.executable, str(script)]]

    def test_config_guards(self):
        for update in [
            {"agent": {"max_requests": 0}},
            {"agent": {"max_requests": 1.5}},
            {"model": {"base_url": "https://example.com/v1"}},
            {"model": {"oops": 1}},
            {"model": {"extra_body": {"messages": []}}},
            {"agent": {"protocol": "single-file"}},
            {"agent": {"fallback_file": "../escape.c"}},
        ]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                load_config(overrides=update)

    def test_confined_paths(self):
        (self.path / "link").symlink_to(self.path.parent)
        for value in ["../escape", "/tmp/escape", "link/file"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                confined(self.path, value)

    def test_workspace_tools_and_limits(self):
        self.config["tools"]["max_file_bytes"] = 5
        tools = Workspace(self.path, self.config)
        tools.action({"type": "write", "path": "a.txt", "content": "12345"}, 10)
        self.assertEqual(tools.action({"type": "read", "path": "a.txt"}, 10)["content"], "12345")
        with self.assertRaises(ValueError):
            tools.action({"type": "write", "path": "b", "content": "123456"}, 10)
        with self.assertRaises(ValueError):
            tools.action({"type": "run", "argv": ["bash", "-c", "true"]}, 10)

    def test_execute_timeout_and_environment(self):
        self.config["tools"]["max_output_chars"] = 10
        result = execute([sys.executable, "-c", "print('a' * 100)"], self.path, self.config)
        self.assertEqual(len(result["output"]), 10)
        self.assertTrue(result["output_truncated"])
        result = execute(
            [sys.executable, "-c", "import time; time.sleep(10)"], self.path, self.config, timeout=0.1
        )
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)

    def test_repair_and_cumulative_resume(self):
        self.validator()
        run = new_run(self.config, "short prompt", self.path / "runs", label="short")
        data = run_session(run, client=FakeClient([reply("bad"), reply("good")]), progress=lambda _: None)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["metrics"]["requests"], 2)
        self.assertEqual(data["metrics"]["total_tokens_known"], 30)
        self.assertEqual(data["metrics"]["repair_cycles"], 1)
        self.assertEqual([v["passed"] for v in data["validation"]], [False, True])
        data = run_session(run, "another check", client=FakeClient([reply()]), progress=lambda _: None)
        self.assertEqual(data["metrics"]["total_tokens_known"], 45)
        self.assertEqual(data["user_followups"], 1)
        self.assertEqual(len(data["sessions"]), 2)
        write_report(load_runs([run]), self.path / "resumed-report")
        report = (self.path / "resumed-report/comparison.md").read_text()
        self.assertIn("another check", report)
        self.assertIn("Календарное время с паузами", report)

    def test_no_validator_never_passed(self):
        run = new_run(self.config, "prompt", self.path)
        data = run_session(run, client=FakeClient([reply()]), progress=lambda _: None)
        self.assertEqual(data["status"], "unverified")

    def test_stalled_replies_and_resume_preserve_cost(self):
        self.validator()
        run = new_run(self.config, "prompt", self.path / "runs")
        data = run_session(
            run, client=FakeClient([reply("bad")] * 3), progress=lambda _: None
        )
        self.assertEqual(data["status"], "stalled")
        self.assertEqual(data["metrics"]["requests"], 3)
        self.assertEqual(data["metrics"]["total_tokens_known"], 45)
        self.assertEqual(len(data["validation"]), 3)
        data = run_session(
            run, "correct the file", client=FakeClient([reply()]), progress=lambda _: None
        )
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["metrics"]["requests"], 4)
        self.assertEqual(data["metrics"]["total_tokens_known"], 60)
        self.assertEqual(data["identical_replies"], 1)
        self.assertEqual(data["user_followups"], 1)

    def test_identical_reply_is_validated_before_stall(self):
        # External validation can become successful even when the reply repeats.
        run = new_run(self.config, "prompt", self.path / "runs")
        checks = [{"passed": False, "checks": []}] * 2 + [{"passed": True, "checks": []}]
        with patch("arpo_lab.agent.validate", side_effect=checks) as validator:
            data = run_session(run, client=FakeClient([reply()] * 3), progress=lambda _: None)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(validator.call_count, 3)
        self.assertEqual(data["metrics"]["requests"], 3)

    def test_protocol_errors_count_and_stop_at_budget(self):
        self.config["agent"]["max_requests"] = 2
        run = new_run(self.config, "prompt", self.path)
        data = run_session(run, client=FakeClient(["bad json", "bad json"]), progress=lambda _: None)
        self.assertEqual(data["status"], "budget_exhausted")
        self.assertEqual(data["metrics"]["requests"], 2)
        self.assertEqual(data["metrics"]["repair_cycles"], 2)

    def test_seed_independence_and_report(self):
        seed = self.path / "seed"
        seed.mkdir()
        (seed / "original.txt").write_text("original")
        first = new_run(self.config, "a", self.path / "runs", seed=seed, label="short")
        second = new_run(self.config, "b", self.path / "runs", seed=seed, label="detailed")
        self.assertEqual(read_json(first / "run.json")["seed_sha256"], tree_hash(seed))
        (first / "workspace/original.txt").write_text("changed")
        self.assertEqual((second / "workspace/original.txt").read_text(), "original")
        for path in [first, second]:
            run_session(path, client=FakeClient([reply()]), progress=lambda _: None)
        runs = load_runs([first, second])
        outputs = write_report(runs, self.path / "report")
        self.assertEqual(len(outputs), 4)
        self.assertEqual(summaries(runs)["short"]["passed"], 0)
        self.assertIn("Не все реализации", (self.path / "report/comparison.md").read_text())
        self.assertIn("Исходный промпт: short", (self.path / "report/comparison.md").read_text())
        self.assertIn("Дополнительные параметры API", (self.path / "report/comparison.md").read_text())

    def test_seed_rejects_symlink(self):
        seed = self.path / "seed"
        seed.mkdir()
        (seed / "link").symlink_to(self.path)
        with self.assertRaises(ValueError):
            new_run(self.config, "a", self.path / "runs", seed=seed)

    def test_parse_schema(self):
        for content in ["[]", '{"done":"yes"}', '{"done":true,"actions":[1]}']:
            with self.assertRaises(ValueError):
                parse_reply(content, 12)

    def test_literal_file_protocol(self):
        text = "```file:solution.c\ndouble f(double x) { return x < 0 ? -x : x; }\n```\n"
        text += '```json\n{"actions":[],"done":true}\n```'
        result = parse_reply(text, 12)
        self.assertEqual(result["actions"][0]["content"], "double f(double x) { return x < 0 ? -x : x; }\n")
        self.assertTrue(result["done"])
        with self.assertRaises(ValueError):
            parse_reply(text, 0)
        text = '````file:README.md\n```c\ncode\n```\n````\n```json\n{"done":true}\n```'
        self.assertEqual(parse_reply(text, 12)["actions"][0]["content"], "```c\ncode\n```\n")

    def test_native_tools_write_and_submit(self):
        self.validator()
        run = new_run(self.config, "prompt", self.path / "runs")
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "w",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"solution.txt","content":"good"}',
                    },
                },
                {"id": "f", "function": {"name": "finish", "arguments": {"summary": "ready"}}},
            ],
        }
        data = run_session(run, client=FakeClient([message]), progress=lambda _: None)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["metrics"]["tool_calls"], 2)
        self.assertEqual([m["tool_call_id"] for m in data["messages"] if m["role"] == "tool"], ["w", "f"])
        bad = {"tool_calls": [{"id": "b", "function": {"name": "write_file", "arguments": "bad json"}}]}
        self.assertEqual(native_reply(bad, 12)["actions"][0]["type"], "invalid")
        self.assertEqual(
            native_reply({"tool_calls": [{"id": "missing"}]}, 12)["actions"][0]["type"], "invalid"
        )

    def test_single_file_format_and_repair(self):
        self.validator()
        (self.path / "validator.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.exit(0 if Path('solution.txt').read_text().strip() == 'good' else 1)"
        )
        self.config = load_config(
            overrides={
                "agent": {"protocol": "single-file", "fallback_file": "solution.txt"},
                "tools": {"sandbox": "off"},
                "validation": self.config["validation"],
            }
        )
        run = new_run(self.config, "prompt", self.path / "runs")
        data = run_session(
            run, client=FakeClient(["```text\nbad\n```", "```text\ngood\n```"]), progress=lambda _: None
        )
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["metrics"]["requests"], 2)
        self.assertEqual(data["metrics"]["repair_cycles"], 1)
        self.assertEqual((run / "workspace/solution.txt").read_text(), "good\n")

    def test_single_file_rejects_alternative_code(self):
        self.assertEqual(single_file_reply("```c\na < b;\n```", "s.c")["actions"][0]["content"], "a < b;\n")
        for text in ["plain code", "```c\na\n```\n```c\nb\n```", "Explanation\n```c\na\n```"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                single_file_reply(text, "s.c")


if __name__ == "__main__":
    unittest.main()
