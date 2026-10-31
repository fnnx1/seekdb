#!/usr/bin/env python3
"""Minimal stand-in for the model gateway used by the ai_function mysqltest suite.

The suite points every endpoint at ``http://0.0.0.0:8000`` and only asserts on
shape, never on model quality: a completion must be non-empty, an embedding must
be 768 wide, and a rerank must return one entry per document.  Without a listener
on that port mysqltest dies at the first call with ``curl error 7`` and the rest
of the suite is never exercised, which is exactly what masks the real diffs.

Supported routes (path suffix match):
  /chat/completions   OpenAI compatible completion
  /embeddings         OpenAI compatible embedding
  /v1/rerank          SiliconFlow style rerank
  /api/generate       Ollama native completion
  /api/embed          Ollama native embedding

Usage:
    python3 mock_ai_service.py serve 8000
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_DIMENSION = 768


def _vector(dimension, seed):
    return [round(((seed + i) % 17) / 17.0, 6) for i in range(dimension)]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Keep the probe log readable: one line per request, no timestamps.
        sys.stderr.write("[mock-ai] " + (fmt % args) + "\n")

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()

        if path.endswith("/chat/completions"):
            prompt = ""
            messages = body.get("messages") or []
            if messages:
                prompt = messages[-1].get("content") or ""
            self._send({
                "id": "mock-completion",
                "object": "chat.completion",
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock reply to " + str(prompt)[:64]},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        elif path.endswith("/embeddings"):
            inputs = body.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            elif not isinstance(inputs, list):
                inputs = [""]
            dimension = body.get("dimensions") or DEFAULT_DIMENSION
            data = []
            for index, text in enumerate(inputs):
                seed = sum(bytearray(str(text).encode("utf-8")))
                data.append({
                    "object": "embedding",
                    "index": index,
                    "embedding": _vector(dimension, seed),
                })
            self._send({
                "object": "list",
                "model": body.get("model", "mock-embedding"),
                "data": data,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        elif path.endswith("/rerank"):
            documents = body.get("documents")
            document_count = len(documents) if isinstance(documents, list) else 0
            results = []
            for index in range(document_count):
                # Descending scores keep the caller supplied order stable, so the
                # fixture's "first id must be 'a'" expectation is well defined.
                results.append({
                    "index": index,
                    "relevance_score": round(1.0 - index * 0.01, 6),
                    "document": documents[index] if isinstance(documents, list) else None,
                })
            self._send({"id": "mock-rerank", "results": results,
                        "usage": {"total_tokens": document_count}})
        elif path.endswith("/api/generate"):
            self._send({
                "model": body.get("model", "mock-model"),
                "response": "mock reply to " + str(body.get("prompt", ""))[:64],
                "done": True,
            })
        elif path.endswith("/api/embed"):
            inputs = body.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            elif not isinstance(inputs, list):
                inputs = [""]
            self._send({
                "model": body.get("model", "mock-embedding"),
                "embeddings": [_vector(DEFAULT_DIMENSION, sum(bytearray(str(text).encode("utf-8"))))
                               for text in inputs],
            })
        else:
            self._send({"error": "unsupported path " + path}, status=404)


def serve(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write("[mock-ai] listening on 0.0.0.0:%d\n" % port)
    sys.stderr.flush()
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("port", type=int, nargs="?", default=8000)
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
