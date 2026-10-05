"""OpenAI-compatible nonstreaming transport; every HTTP attempt is recorded."""

import json
import time
import urllib.error
import urllib.request

from .config import api_key
from .protocol import qwen_xml_reply, schemas
from .storage import utc_now


class APIError(RuntimeError):
    pass


def usage_counts(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (prompt, completion)):
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class Client:
    def __init__(self, config):
        self.config = config
        # Do not let host HTTP_PROXY silently redirect local prompts.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.tokenizer = None

    def models(self):
        request = urllib.request.Request(self.config["model"]["base_url"].rstrip("/") + "/models")
        if key := api_key(self.config):
            request.add_header("Authorization", "Bearer " + key)
        with self.opener.open(request, timeout=5) as response:
            return json.load(response)

    def complete(self, messages, record, can_request=lambda: True, remaining_s=lambda: 300):
        model = self.config["model"]
        body = {
            "model": model["name"],
            "messages": messages,
            "stream": False,
            "temperature": model["temperature"],
            "top_p": model["top_p"],
            "max_tokens": model["max_tokens"],
            "seed": model["seed"],
            **model["extra_body"],
        }
        xml = self.config["agent"]["protocol"] == "qwen-xml"
        native = self.config["agent"]["protocol"] in {"native", "qwen-xml"}
        if native:
            body["tools"] = schemas(self.config)
        if xml:
            # MLX 0.32 exposes original token IDs in logprobs, before its tool parser.
            # Exact usage still comes from the server, not from decoding the text.
            body["logprobs"] = True
            if self.tokenizer is None:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model["name"], local_files_only=True)
        last_error = ""
        for retry in range(model["max_retries"] + 1):
            if not can_request():
                raise APIError("Request, token or time budget exhausted")
            started = time.perf_counter()
            event = {
                "started_at": utc_now(),
                "retry": retry,
                "request": body,
                "usage": None,
                "ok": False,
                "response": None,
            }
            retryable = False
            try:
                request = urllib.request.Request(
                    model["base_url"].rstrip("/") + "/chat/completions",
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                if key := api_key(self.config):
                    request.add_header("Authorization", "Bearer " + key)
                with self.opener.open(
                    request, timeout=max(0.1, min(model["timeout_s"], remaining_s()))
                ) as http:
                    event["http_status"] = http.status
                    response = json.load(http)
                event["response"] = response
                event["usage"] = usage_counts(response)
                message = response["choices"][0]["message"]
                if xml:
                    entries = response["choices"][0].get("logprobs", {}).get("content", [])
                    ids = [entry.get("id") for entry in entries]
                    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
                        raise APIError("qwen-xml requires MLX logprobs with raw token IDs")
                    message = qwen_xml_reply(self.tokenizer.decode(ids, skip_special_tokens=False))
                content = message.get("content")
                if not isinstance(content, str) and not (native and message.get("tool_calls")):
                    raise APIError("Expected textual message.content; native tool calls are not used")
                if event["usage"] is None and model["require_usage"]:
                    raise APIError("Server returned no valid usage; exact token accounting is required")
                event["ok"] = True
                return message if native else content
            except urllib.error.HTTPError as exc:
                event["http_status"] = exc.code
                # Response body may contain credentials from upstream; keep status only.
                last_error = f"HTTP {exc.code}"
                retryable = exc.code in {408, 429} or exc.code >= 500
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = f"{type(exc).__name__}: transport failure"
                retryable = True
            except (KeyError, IndexError, TypeError, ValueError, APIError) as exc:
                last_error = f"Invalid response: {exc}"
            finally:
                event["duration_s"] = time.perf_counter() - started
                if not event["ok"] or last_error:
                    event["error"] = last_error
                record(event)
            if not retryable or retry == model["max_retries"]:
                break
            if remaining_s() <= model["retry_delay_s"]:
                break
            time.sleep(model["retry_delay_s"])
            last_error = ""
        raise APIError(last_error)
