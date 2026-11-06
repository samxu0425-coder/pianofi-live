"""Live piano roll: runs the engine and streams its note events to the browser.

    .venv\\Scripts\\python.exe ui\\server.py test_audio\\16k\\song.wav
    .venv\\Scripts\\python.exe ui\\server.py --input loopback

Then open http://localhost:8000 and press Play. With a wav, the page plays the song.
With --input loopback (what the PC is playing) or mic, play the music yourself.
The page is served over HTTP on --port, and note events go over a WebSocket on --port + 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.asyncio.server import broadcast, serve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from sweep import engine_env, find_exe


class Files(BaseHTTPRequestHandler):
    """Serves the page, and the song when there is one."""

    def __init__(self, wav: Path | None, *args, **kwargs):
        self.files = {"/": (ROOT / "ui" / "index.html", "text/html; charset=utf-8")}
        if wav:
            self.files["/song.wav"] = (wav, "audio/wav")
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path not in self.files:
            self.send_error(404)
            return
        path, content_type = self.files[self.path]
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # Keep the console for the engine's logs.


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path, nargs="?", help="16 kHz wav from py/prep_audio.py (file input only)")
    parser.add_argument("--input", choices=["file", "mic", "loopback"], default="file",
                        help="loopback = what the PC is playing")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=16, help="frames per pass, 16 ms each")
    parser.add_argument("--delay", type=int, default=16, help="delay in frames, 16 ms each")
    args = parser.parse_args()
    if (args.input == "file") != (args.wav is not None):
        parser.error("give either a wav file or --input mic|loopback")
    wav = args.wav.resolve() if args.wav else None
    source = ["--wav", str(wav)] if wav else ["--input", args.input]

    clients = set()
    engine = None     # The running engine process, if any.
    stopped = False   # True when the page asked for the stop, so it isn't shown as a failure.

    async def run_engine():
        nonlocal engine, stopped
        stopped = False
        engine = await asyncio.create_subprocess_exec(
            str(find_exe()), *source, "--stride", str(args.stride), "--delay", str(args.delay),
            "--events", "--out", str(ROOT / "runs" / "ui"),
            cwd=ROOT, env=engine_env(), stdout=asyncio.subprocess.PIPE,  # stderr stays on this console
        )
        async for line in engine.stdout:
            broadcast(clients, line.decode().strip())
        code = await engine.wait()
        engine = None
        broadcast(clients, json.dumps({"type": "end", "code": code, "stopped": stopped}))

    async def on_client(ws):
        nonlocal stopped
        clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "mode", "input": args.input, "running": engine is not None}))
            async for message in ws:
                if message == "play" and engine is None:
                    asyncio.create_task(run_engine())
                elif message == "stop" and engine is not None:
                    stopped = True
                    engine.terminate()
        finally:
            clients.discard(ws)

    http_server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Files, wav))
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{args.port} and press Play. Ctrl+C to quit.", flush=True)
    try:
        async with serve(on_client, "127.0.0.1", args.port + 1):
            await asyncio.Future()  # Run until Ctrl+C.
    finally:
        if engine is not None:
            engine.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
