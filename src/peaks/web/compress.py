"""Gzip for the text Peaks sends (JSON, JS, CSS, HTML) — never for video,
images or anything sent in ranges, so seeking in a stream is untouched.

Starlette's GZipMiddleware compresses any type; a ranged mp4 must not be."""

from __future__ import annotations

import gzip

TEXT = ("application/json", "text/", "application/javascript", "image/svg+xml", "application/manifest+json")
MIN_SIZE = 1024
MAX_SIZE = 32 * 1024 * 1024          # bigger than this goes out as it is


class TextGzip:
    def __init__(self, app, minimum_size: int = MIN_SIZE):
        self.app, self.minimum = app, minimum_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        if b"gzip" not in headers.get(b"accept-encoding", b"") or b"range" in headers:
            return await self.app(scope, receive, send)

        start = None
        body = bytearray()
        passthrough = False

        async def wrapped(message):
            nonlocal start, passthrough
            if passthrough:
                return await send(message)
            if message["type"] == "http.response.start":
                h = {k.lower(): v for k, v in message.get("headers", [])}
                ctype = h.get(b"content-type", b"").decode("latin-1")
                if (b"content-encoding" in h or message.get("status", 200) == 206
                        or not ctype.startswith(TEXT)):
                    passthrough = True
                    return await send(message)
                start = message
                return None
            if message["type"] != "http.response.body":
                return await send(message)
            body.extend(message.get("body", b""))
            if message.get("more_body") and len(body) <= MAX_SIZE:
                return None
            if message.get("more_body") or len(body) < self.minimum:
                passthrough = True                    # too big to hold, or too small to bother
                await send(start)
                return await send({"type": "http.response.body", "body": bytes(body),
                                   "more_body": message.get("more_body", False)})
            data = gzip.compress(bytes(body), compresslevel=5)
            hs = [(k, v) for k, v in start.get("headers", []) if k.lower() not in (b"content-length",)]
            hs += [(b"content-encoding", b"gzip"), (b"content-length", str(len(data)).encode()),
                   (b"vary", b"Accept-Encoding")]
            await send({**start, "headers": hs})
            await send({"type": "http.response.body", "body": data, "more_body": False})

        await self.app(scope, receive, wrapped)
