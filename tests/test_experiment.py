import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arpo_lab.config import load_config
from arpo_lab.experiment import experiment
from arpo_lab.report import load_runs
from arpo_lab.storage import read_json


class ExperimentTests(unittest.TestCase):
    def test_alternate_order_and_independent_histories(self):
        observed = []

        class Client:
            def __init__(self, config):
                pass

            def complete(self, messages, record, can_request, remaining_s):
                observed.append([m.copy() for m in messages])
                text = messages[1]["content"]
                record(
                    {
                        "ok": True,
                        "duration_s": 0.01,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    }
                )
                return json.dumps(
                    {"done": True, "actions": [{"type": "write", "path": "solution.txt", "content": text}]}
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short, detailed = root / "short.txt", root / "detailed.txt"
            short.write_text("SHORT")
            detailed.write_text("DETAILED")
            config = load_config(overrides={"experiment": {"repeats": 3, "warmup": False}})
            with patch("arpo_lab.agent.Client", Client):
                output = experiment(config, short, detailed, root / "runs", progress=lambda _: None)
            manifest = read_json(output / "experiment.json")
            self.assertEqual(
                [r["label"] for r in manifest["runs"]],
                ["short", "detailed", "detailed", "short", "short", "detailed"],
            )
            self.assertEqual(len({r["path"] for r in manifest["runs"]}), 6)
            self.assertTrue(all(len(history) == 2 for history in observed))
            runs = load_runs([output])
            self.assertEqual(len(runs), 6)
            self.assertTrue(all(r["metrics"]["total_tokens_known"] == 7 for r in runs))
            for entry in manifest["runs"]:
                content = (output / entry["path"] / "workspace/solution.txt").read_text()
                self.assertEqual(content, "SHORT" if entry["label"] == "short" else "DETAILED")


if __name__ == "__main__":
    unittest.main()
