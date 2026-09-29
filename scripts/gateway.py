#!/usr/bin/env python3
"""Small streaming gateway that pools heterogeneous Dynamo deployments."""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


def load_backends(path: Path) -> list[dict]:
    config = json.loads(path.read_text())
    return [
        {**backend, "active": 0, "requests": 0, "failures": 0}
        for backend in config["gateway"]["backends"]
    ]


def create_app(backends: list[dict]) -> FastAPI:
    client: httpx.AsyncClient
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client
        client = httpx.AsyncClient(timeout=None)
        yield
        await client.aclose()

    app = FastAPI(title="Heterogeneous Inference Gateway", lifespan=lifespan)

    async def choose_backend() -> dict:
        async with lock:
            backend = min(backends, key=lambda item: (item["active"] + 1) / item["weight"])
            backend["active"] += 1
            backend["requests"] += 1
            return backend

    async def release(backend: dict, response: httpx.Response):
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except Exception:
            backend["failures"] += 1
            raise
        finally:
            await response.aclose()
            async with lock:
                backend["active"] -= 1

    @app.get("/health")
    async def health():
        checks = await asyncio.gather(
            *(client.get(f"{backend['url']}/health", timeout=3) for backend in backends),
            return_exceptions=True,
        )
        status = {
            backend["name"]: not isinstance(check, Exception) and check.is_success
            for backend, check in zip(backends, checks)
        }
        return JSONResponse(status, status_code=200 if all(status.values()) else 503)

    @app.get("/gateway/status")
    async def gateway_status():
        return [{key: value for key, value in backend.items() if key != "url"} for backend in backends]

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        backend = await choose_backend()
        headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
        try:
            outbound = client.build_request(
                request.method,
                f"{backend['url']}/{path}",
                params=request.query_params,
                headers=headers,
                content=await request.body(),
            )
            response = await client.send(outbound, stream=True)
        except Exception as error:
            async with lock:
                backend["active"] -= 1
                backend["failures"] += 1
            raise HTTPException(status_code=502, detail=f"{backend['name']}: {error}") from error
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        response_headers["x-inference-backend"] = backend["name"]
        return StreamingResponse(
            release(backend, response),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    gateway = config["gateway"]
    uvicorn.run(create_app(load_backends(args.config)), host="0.0.0.0", port=gateway["port"])


if __name__ == "__main__":
    main()
