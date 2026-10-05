import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from arpo_lab.api import APIError, Client, usage_counts
from arpo_lab.config import load_config


class APITests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                status, response = outer.responses.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = load_config(
            overrides={
                "model": {"base_url": f"http://127.0.0.1:{self.server.server_port}/v1", "retry_delay_s": 0}
            }
        )
        self.events = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def response(self, usage=True):
        value = {"choices": [{"message": {"content": "OK"}}]}
        if usage:
            value["usage"] = {"prompt_tokens": 7, "completion_tokens": 2}
        return value

    def test_request_and_exact_usage(self):
        self.responses = [(200, self.response())]
        value = Client(self.config).complete([{"role": "user", "content": "hi"}], self.events.append)
        self.assertEqual(value["content"], "OK")
        self.assertFalse(self.requests[0]["stream"])
        self.assertEqual(self.events[0]["usage"]["total_tokens"], 9)
        self.assertIn("tools", self.requests[0])

    def test_native_response_without_text(self):
        response = self.response()
        response["choices"][0]["message"] = {
            "content": None,
            "tool_calls": [{"id": "f", "function": {"name": "finish", "arguments": '{"summary":"done"}'}}],
        }
        self.responses = [(200, response)]
        value = Client(self.config).complete([], self.events.append)
        self.assertEqual(value["tool_calls"][0]["id"], "f")

    def test_qwen_xml_uses_raw_ids_and_keeps_server_usage(self):
        self.config["agent"]["protocol"] = "qwen-xml"
        client = Client(self.config)

        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                self.ids = ids
                return (
                    "<function=read_file><parameter=path>SharedFiles/api.h</parameter></function><|im_end|>"
                )

        client.tokenizer = Tokenizer()
        response = self.response()
        response["choices"][0]["logprobs"] = {"content": [{"id": 123}, {"id": 456}]}
        self.responses = [(200, response)]
        message = client.complete([], self.events.append)
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": "SharedFiles/api.h"}
        )
        self.assertEqual(client.tokenizer.ids, [123, 456])
        self.assertTrue(self.requests[0]["logprobs"])
        self.assertEqual(self.events[0]["usage"]["total_tokens"], 9)
        self.assertEqual(self.events[0]["response"]["choices"][0]["message"]["content"], "OK")

    def test_qwen_xml_refuses_missing_ids_without_losing_usage(self):
        self.config["agent"]["protocol"] = "qwen-xml"
        client = Client(self.config)
        client.tokenizer = object()
        self.responses = [(200, self.response())]
        with self.assertRaisesRegex(APIError, "raw token IDs"):
            client.complete([], self.events.append)
        self.assertEqual(self.events[0]["usage"]["total_tokens"], 9)
        self.assertFalse(self.events[0]["ok"])

    def test_single_file_omits_tools(self):
        self.config = load_config(
            overrides={
                "model": {"base_url": self.config["model"]["base_url"]},
                "agent": {"protocol": "single-file", "fallback_file": "solution.c"},
            }
        )
        self.responses = [(200, self.response())]
        self.assertEqual(Client(self.config).complete([], self.events.append), "OK")
        self.assertNotIn("tools", self.requests[0])

    def test_files_protocol_omits_tools_and_keeps_usage(self):
        self.config["agent"]["protocol"] = "files"
        response = self.response()
        response["choices"][0]["message"]["content"] = "```file:test.c\nint x = 1;\n```"
        self.responses = [(200, response)]
        self.assertIn("file:test.c", Client(self.config).complete([], self.events.append))
        self.assertNotIn("tools", self.requests[0])
        self.assertEqual(self.events[0]["usage"]["total_tokens"], 9)

    def test_retry_counts_both_attempts(self):
        self.responses = [(503, {}), (200, self.response())]
        Client(self.config).complete([], self.events.append)
        self.assertEqual(len(self.events), 2)
        self.assertFalse(self.events[0]["ok"])
        self.assertIsNone(self.events[0]["usage"])
        self.assertTrue(self.events[1]["ok"])

    def test_400_is_not_retried(self):
        self.responses = [(400, {})]
        with self.assertRaises(APIError):
            Client(self.config).complete([], self.events.append)
        self.assertEqual(len(self.events), 1)

    def test_usage_missing_is_not_invented(self):
        self.responses = [(200, self.response(False))]
        with self.assertRaises(APIError):
            Client(self.config).complete([], self.events.append)
        self.assertIsNone(self.events[0]["usage"])
        self.assertFalse(self.events[0]["ok"])
        self.assertEqual(len(self.requests), 1)

    def test_budget_prevents_retry(self):
        self.responses = [(503, {})]
        with self.assertRaises(APIError):
            Client(self.config).complete([], self.events.append, can_request=lambda: len(self.events) == 0)
        self.assertEqual(len(self.events), 1)

    def test_invalid_usage(self):
        self.assertIsNone(usage_counts({"usage": {"prompt_tokens": True, "completion_tokens": 1}}))
        self.assertIsNone(usage_counts({"usage": {"prompt_tokens": -1, "completion_tokens": 1}}))


if __name__ == "__main__":
    unittest.main()
