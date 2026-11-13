"""Live piano roll: runs the engine and streams its note events to the browser.

    .venv\\Scripts\\python.exe ui\\server.py                      songs from test_audio\\16k, pick one on the page
    .venv\\Scripts\\python.exe ui\\server.py test_audio\\16k\\song.wav   same, with that song selected
    .venv\\Scripts\\python.exe ui\\server.py --input loopback

Then open http://localhost:8000 and press Play. With a song, the page plays it.
With --input loopback (what the PC is playing) or mic, play the music yourself.
Songs dropped on the page are decoded and resampled to 16 kHz mono by the browser,
uploaded, and saved in the songs folder (--songs, default test_audio\\16k).
A page can also stream its own audio (the microphone, or a tab's sound): the browser sends
16 kHz float32 samples over the WebSocket and the engine reads them from stdin.
Every open page is its own session with its own engine and song, up to --max-sessions
at once (default 2, what one T4 keeps up with); closing the page stops its engine.
The page is served over HTTP on --port, and note events go over a WebSocket on --port + 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import soundfile as sf
from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parent.parent

# On this PC the engine is the Release build and needs the CUDA DLLs from torch on PATH
# (eval/sweep.py knows both). In the service image PIANOFI_ENGINE names it and the
# CUDA libraries are already on the library path.
if "PIANOFI_ENGINE" in os.environ:
    WORK = Path.cwd()  # /work in the container: models/, songs/ and runs/ are mounted there

    def find_exe() -> Path:
        return Path(os.environ["PIANOFI_ENGINE"])

    def engine_env() -> dict[str, str]:
        return dict(os.environ)
else:
    WORK = ROOT
    sys.path.insert(0, str(ROOT / "eval"))
    from sweep import engine_env, find_exe

# How to ask the engine to finish. On Windows a signal can only be sent to a process
# that is in its own group, and then only Ctrl+Break (not Ctrl+C) reaches it.
WINDOWS = sys.platform == "win32"
STOP_SIGNAL = signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT
NEW_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
STOP_TIMEOUT_SEC = 15  # the engine still has to flush its last notes
ENGINE_RATE = 16000
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 50 minutes of 16 kHz float32


class Songs:
    """The wavs the engine can play: everything in the songs folder, plus uploads."""

    def __init__(self, folder: Path, extra: Path | None):
        self.folder = folder
        self.extra = extra  # a song given on the command line, wherever it is
        self.folder.mkdir(parents=True, exist_ok=True)

    def paths(self) -> dict[str, Path]:
        songs = {p.name: p for p in sorted(self.folder.glob("*.wav"))}
        if self.extra:
            songs[self.extra.name] = self.extra
        return songs

    def names(self) -> list[str]:
        return list(self.paths())

    def save(self, name: str, data: bytes) -> str:
        """Store an uploaded wav. It must already be what the engine wants: 16 kHz mono."""
        name = re.sub(r"[^\w .()\-]+", "_", Path(name).stem).strip() or "song"
        path = self.folder / f"{name}.wav"
        path.write_bytes(data)
        try:
            info = sf.info(str(path))
            if info.samplerate != ENGINE_RATE or info.channels != 1:
                raise ValueError(f"need {ENGINE_RATE} Hz mono, got {info.samplerate} Hz, {info.channels} channels")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path.name


class Files(BaseHTTPRequestHandler):
    """Serves the page, the songs, and takes uploads (from the admin only, when one is set)."""

    def __init__(self, songs: Songs, admin: str, status, *args, **kwargs):
        self.songs = songs
        self.admin = admin
        self.status = status  # () -> dict, for /status (the GPU box's idle check reads it)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_bytes((ROOT / "ui" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/status":
            self.send_bytes(json.dumps(self.status()).encode(), "application/json")
        elif url.path.startswith("/songs/"):
            path = self.songs.paths().get(unquote(url.path[len("/songs/"):]))
            if path is None:
                self.send_error(404)
            else:
                self.send_bytes(path.read_bytes(), "audio/wav")
        else:
            self.send_error(404)

    def do_HEAD(self):
        # The offline page asks "is the app back?" with HEAD / every 10 s while the box starts.
        if urlparse(self.path).path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/upload":
            self.send_error(404)
            return
        name = parse_qs(url.query).get("name", ["song"])[0]
        length = int(self.headers.get("Content-Length", 0))
        if not is_admin(self.headers.get("X-User"), self.admin):
            # Read the body before answering, or the client sees a reset instead of the reason.
            remaining = min(length, MAX_UPLOAD_BYTES)
            while remaining > 0:
                remaining -= len(self.rfile.read(min(1 << 20, remaining)) or b"\0")
            self.send_error(403, f"only {self.admin} can add songs")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_error(413, "upload too large" if length > 0 else "empty upload")
            return
        try:
            saved = self.songs.save(name, self.rfile.read(length))
        except Exception as e:
            self.send_error(400, str(e))
            return
        self.send_bytes(json.dumps({"name": saved}).encode(), "application/json")

    def send_bytes(self, data: bytes, content_type: str):
        # Browsers can only seek in audio served with byte ranges (used when the page
        # jumps the song forward to where the engine is).
        start, end = 0, len(data) - 1
        status = 200
        if match := re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")):
            status = 206
            if match[1]:
                start, end = int(match[1]), int(match[2]) if match[2] else end
            else:  # "bytes=-N": the last N bytes
                start = max(len(data) - int(match[2]), 0)
            end = min(end, len(data) - 1)
            if start > end:
                self.send_error(416)
                return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def log_message(self, *args):
        pass  # Keep the console for the engine's logs.


def is_admin(user, admin: str) -> bool:
    """Who may add songs. Behind the front, Caddy puts the login name in X-User (a client
    cannot set it). With no --admin, as on your own PC, everyone may."""
    return not admin or user == admin


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path, nargs="?", help="16 kHz wav to select first (file input only)")
    parser.add_argument("--songs", type=Path, default=ROOT / "test_audio" / "16k",
                        help="folder of 16 kHz wavs to choose from; uploads are saved there")
    parser.add_argument("--input", choices=["file", "mic", "loopback"], default="file",
                        help="loopback = what the PC is playing")
    parser.add_argument("--device", default="",
                        help="part of a device name, for mic or loopback (pianofi_live --list-devices shows them)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines (behind a proxy)")
    parser.add_argument("--stride", type=int, default=16, help="frames per pass, 16 ms each")
    parser.add_argument("--delay", type=int, default=16, help="delay in frames, 16 ms each")
    parser.add_argument("--max-sessions", type=int, default=2,
                        help="engines allowed at once, one per page that pressed Play (a T4 keeps up with 2, 4 with --fp16)")
    parser.add_argument("--fp16", action="store_true",
                        help="run the engines on the half-precision graphs (models/*_fp16.onnx, py/export_fp16.py)")
    parser.add_argument("--admin", default="",
                        help="login name (X-User, set by the front's Caddy) that may add songs; default: everyone")
    args = parser.parse_args()
    if args.input != "file" and args.wav is not None:
        parser.error("a wav goes with file input only")
    live = args.input != "file"
    songs = Songs(args.songs.resolve(), args.wav.resolve() if args.wav else None)
    if not live and not songs.names():
        parser.error(f"no .wav files in {songs.folder}; run py/prep_audio.py or drop a song on the page")
    selected = args.wav.name if args.wav else (songs.names()[0] if not live else "")

    # Every connected page is its own session with its own engine, song and run folder.
    # Closing the page stops its engine, so a session never outlives its user.
    sessions: dict[object, Session] = {}
    next_id = 0

    class Session:
        def __init__(self, ws, sid: int):
            self.ws = ws
            self.id = sid
            self.engine = None     # The running engine process, if any.
            self.stopped = False   # True when the page asked for the stop, so it isn't shown as a failure.
            self.song = selected
            request = getattr(ws, "request", None)  # websockets >= 13
            self.user = request.headers.get("X-User", "") if request is not None else ""

        def mode(self) -> str:
            return json.dumps({"type": "mode", "input": args.input, "running": self.engine is not None,
                               "songs": songs.names(), "song": self.song, "session": self.id,
                               "user": self.user, "can_upload": is_admin(self.user, args.admin)})

        async def send(self, text: str):
            try:
                await self.ws.send(text)
            except Exception:
                pass  # the page went away; the disconnect handler cleans up

        async def run_engine(self, song: str, browser_audio: bool, start: float = 0.0):
            self.stopped = False
            self.song = song
            if browser_audio:  # the page streams 16 kHz float32 samples; they go to the engine's stdin
                source = ["--input", "stdin"]
            elif live:
                source = ["--input", args.input] + (["--device", args.device] if args.device else [])
            else:
                source = ["--wav", str(songs.paths()[song])]
                if start > 0:  # resuming a stopped song from where it stopped
                    source += ["--start-seconds", f"{start:.3f}"]
            self.engine = await asyncio.create_subprocess_exec(
                str(find_exe()), *source, "--stride", str(args.stride), "--delay", str(args.delay),
                "--events", "--max-engines", str(args.max_sessions), *(["--fp16"] if args.fp16 else []),
                "--out", str(WORK / "runs" / "ui" / f"session-{self.id}"),
                cwd=WORK, env=engine_env(), stdout=asyncio.subprocess.PIPE,  # stderr stays on this console
                stdin=asyncio.subprocess.PIPE if browser_audio else None,
                creationflags=NEW_GROUP,
            )
            await self.send(json.dumps({"type": "song", "song": song if not browser_audio else ""}))
            async for line in self.engine.stdout:
                await self.send(line.decode().strip())
            code = await self.engine.wait()
            self.engine = None
            await self.send(json.dumps({"type": "end", "code": code, "stopped": self.stopped}))

        async def stop_engine(self):
            """Ask the engine to finish, so it writes its notes. Kill it if it won't."""
            process = self.engine
            if process is None or self.stopped:
                return
            self.stopped = True
            try:
                if process.stdin is not None:  # browser audio: end of stream ends the engine
                    process.stdin.close()
                else:
                    process.send_signal(STOP_SIGNAL)
                await asyncio.wait_for(process.wait(), STOP_TIMEOUT_SEC)
            except (OSError, ValueError, asyncio.TimeoutError):
                if process.returncode is None:
                    process.terminate()

    def running_count() -> int:
        return sum(1 for s in sessions.values() if s.engine is not None)

    async def on_client(ws):
        nonlocal next_id
        next_id += 1
        session = sessions[ws] = Session(ws, next_id)
        try:
            await session.send(session.mode())
            async for message in ws:
                if isinstance(message, bytes):  # audio from the browser: straight to the engine
                    engine = session.engine
                    if engine is not None and engine.stdin is not None and not engine.stdin.is_closing():
                        engine.stdin.write(message)
                        try:
                            await engine.stdin.drain()
                        except (ConnectionError, BrokenPipeError):
                            pass  # the engine ended; its "end" event follows
                    continue
                try:
                    request = json.loads(message)
                except ValueError:
                    request = {"type": message}
                kind = request.get("type")
                if kind == "play" and session.engine is None:
                    browser_audio = request.get("source") == "browser"
                    song = request.get("song") or selected
                    if not browser_audio and not live and song not in songs.paths():
                        continue
                    if running_count() >= args.max_sessions:
                        await session.send(json.dumps({"type": "busy", "max": args.max_sessions}))
                        continue
                    try:
                        start = max(0.0, float(request.get("start", 0)))
                    except (TypeError, ValueError):
                        start = 0.0
                    asyncio.create_task(session.run_engine(song, browser_audio, start))
                elif kind == "stop":
                    asyncio.create_task(session.stop_engine())
                elif kind == "songs":  # after an upload: every page gets the new list
                    for other in list(sessions.values()):
                        await other.send(other.mode())
        finally:
            sessions.pop(ws, None)
            if session.engine is not None:  # the page is gone: free the GPU
                await session.stop_engine()

    def status() -> dict:
        return {"running": running_count(), "pages": len(sessions)}

    http_server = ThreadingHTTPServer((args.host, args.port), partial(Files, songs, args.admin, status))
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{args.port} and press Play. Ctrl+C to quit.", flush=True)
    try:
        async with serve(on_client, args.host, args.port + 1):
            await asyncio.Future()  # Run until Ctrl+C.
    finally:
        for session in list(sessions.values()):
            if session.engine is not None:
                session.engine.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
