"""Exercise the direct Dynamo HTTP API; save capability checks, not quality scores."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def check_api(base, model, concurrency=8):
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        return urllib.request.urlopen(
            urllib.request.Request(base.rstrip('/') + path, data=data,
                                   headers={'Content-Type': 'application/json'}),
            timeout=180,
        )

    def post(path, payload):
        with request(path, payload) as response:
            return json.load(response)

    checks = {}
    with request('/health') as response:
        checks['health'] = response.status == 200
    with request('/v1/models') as response:
        assert model in {m['id'] for m in json.load(response)['data']}
    checks['model_discovery'] = True
    common = {'model': model, 'temperature': 0, 'max_tokens': 128,
              'chat_template_kwargs': {'enable_thinking': False}}
    messages = [{'role': 'user', 'content': 'Reply briefly: what is two plus three?'}]
    chat = {**common, 'messages': messages}
    completion = post('/v1/completions', {'model': model, 'prompt': 'Two plus three is',
                                          'max_tokens': 16, 'temperature': 0})
    assert completion['choices'] and completion['usage']['completion_tokens'] > 0
    checks['completion'] = True
    first = post('/v1/chat/completions', chat)
    assert first['choices'][0]['message'].get('content')
    checks['chat'] = True
    with request('/v1/chat/completions', {**chat, 'stream': True}) as response:
        assert 'text/event-stream' in response.headers.get('Content-Type', '')
        chunks, done = 0, False
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            assert 'error' not in event, event
            chunks += bool(event.get('choices'))
        assert done and chunks > 0
    checks['sse_stream'] = {'chunks': chunks, 'done': done}
    repeat = post('/v1/chat/completions', chat)
    assert repeat['choices'][0]['message'].get('content')
    checks['repeated_prefix_request'] = 'passed; cache reuse requires engine metrics'
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(lambda _: post('/v1/chat/completions', chat), range(concurrency)))
    assert all(r['choices'][0]['message'].get('content') for r in replies)
    checks['concurrent_chat'] = len(replies)
    tool = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'},
             'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}
    history = [{'role': 'user', 'content': 'Use the add tool with a=2 and b=3, then state its result.'}]
    reply = post('/v1/chat/completions', {**common, 'messages': history,
                                         'tools': [tool], 'tool_choice': 'auto'})
    message = reply['choices'][0]['message']
    calls = message.get('tool_calls') or []
    assert calls, 'model did not emit a tool call'
    history.append(message)
    for call in calls:
        assert call['function']['name'] == 'add'
        args = json.loads(call['function']['arguments'])
        assert args == {'a': 2, 'b': 3}, args
        history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': '5'})
    final = post('/v1/chat/completions', {**common, 'messages': history,
                                        'tools': [tool], 'tool_choice': 'none'})
    assert final['choices'][0]['message'].get('content')
    checks['tool_call_roundtrip'] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Root URL without /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'model': args.model, 'base_url': args.base_url, 'started_unix': time.time()}
    try:
        result['checks'] = check_api(args.base_url, args.model)
        result['passed'] = True
    except Exception as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
