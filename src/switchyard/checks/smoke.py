"""Exercise the direct Dynamo HTTP API; save capability checks, not quality scores."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_TIMEOUT = 180
CHAT = [{"role": "user", "content": "Reply briefly: what is two plus three?"}]
ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


def _opener(base):
    root = base.rstrip("/")

    def open_path(path, payload=None, timeout=DEFAULT_TIMEOUT):
        data = None if payload is None else json.dumps(payload).encode()
        return urllib.request.urlopen(
            urllib.request.Request(
                root + path, data=data, headers={"Content-Type": "application/json"}
            ),
            timeout=timeout,
        )

    return open_path


def check_api(base, model, concurrency=8):
    """Return a dict of named checks; raises on the first hard failure."""
    open_path = _opener(base)

    def post(path, payload):
        with open_path(path, payload) as response:
            return json.load(response)

    checks = {}
    with open_path("/health") as response:
        checks["health"] = response.status == 200
    with open_path("/v1/models") as response:
        assert model in {item["id"] for item in json.load(response)["data"]}
    checks["model_discovery"] = True

    common = {
        "model": model,
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    chat = {**common, "messages": CHAT}

    completion = post(
        "/v1/completions",
        {
            "model": model,
            "prompt": "Two plus three is",
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    assert completion["choices"] and completion["usage"]["completion_tokens"] > 0
    checks["completion"] = True

    first = post("/v1/chat/completions", chat)
    assert first["choices"][0]["message"].get("content")
    checks["chat"] = True

    with open_path("/v1/chat/completions", {**chat, "stream": True}) as response:
        assert "text/event-stream" in response.headers.get("Content-Type", "")
        chunks, done = 0, False
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event, event
            chunks += bool(event.get("choices"))
        assert done and chunks > 0
    checks["sse_stream"] = {"chunks": chunks, "done": done}

    repeat = post("/v1/chat/completions", chat)
    assert repeat["choices"][0]["message"].get("content")
    checks["repeated_prefix_request"] = "passed; cache reuse requires engine metrics"

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(
            pool.map(lambda _: post("/v1/chat/completions", chat), range(concurrency))
        )
    assert all(reply["choices"][0]["message"].get("content") for reply in replies)
    checks["concurrent_chat"] = len(replies)

    history = [
        {
            "role": "user",
            "content": "Use the add tool with a=2 and b=3, then state its result.",
        }
    ]
    reply = post(
        "/v1/chat/completions",
        {**common, "messages": history, "tools": [ADD_TOOL], "tool_choice": "auto"},
    )
    message = reply["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert calls, "model did not emit a tool call"
    history.append(message)
    for call in calls:
        assert call["function"]["name"] == "add"
        arguments = json.loads(call["function"]["arguments"])
        assert arguments == {"a": 2, "b": 3}, arguments
        history.append({"role": "tool", "tool_call_id": call["id"], "content": "5"})
    final = post(
        "/v1/chat/completions",
        {**common, "messages": history, "tools": [ADD_TOOL], "tool_choice": "none"},
    )
    assert final["choices"][0]["message"].get("content")
    checks["tool_call_roundtrip"] = True
    return checks


def run(base, model, output, concurrency=8):
    """Run every check, write a JSON report, and return the report."""
    result = {"model": model, "base_url": base, "started_unix": time.time()}
    try:
        result["checks"] = check_api(base, model, concurrency)
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
