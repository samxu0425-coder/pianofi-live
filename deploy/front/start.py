#!/usr/bin/env python3
"""The "Start the engine" button. Runs on the front machine, behind Caddy on 127.0.0.1:9000.

Caddy has already checked the password and put the login name in X-User. The admin
(AUTH_USER) may start the GPU box as often as needed; anyone else gets GUEST_STARTS_PER_DAY
starts a day (UTC days), counted in /opt/pianofi/data/starts.json. The start itself is the
AWS CLI with the machine's role, which allows exactly this one instance (infra/deploy/iam.tf).

  GET  /start   {"user", "admin", "left", "state"}      left is null for the admin
  POST /start   starts the box if it is stopped; 429 when the day's starts are used up
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

INSTANCE = os.environ["INSTANCE"]
ADMIN = os.environ["AUTH_USER"]
LIMIT = int(os.environ.get("GUEST_STARTS_PER_DAY", "3"))
COUNTS = Path("/opt/pianofi/data/starts.json")


def aws(*args) -> dict:
    result = subprocess.run(["aws", "ec2", *args, "--output", "json"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "aws failed")
    return json.loads(result.stdout)


def state() -> str:
    """pending, running, stopping, stopped (or shutting-down, terminated)."""
    return aws("describe-instances", "--instance-ids", INSTANCE)["Reservations"][0]["Instances"][0]["State"]["Name"]


def start_instance() -> None:
    """Start the box. Right after a stop the instance reads "stopped" a few seconds before its
    spot request is ready to start again (IncorrectSpotRequestState), so try a few times."""
    for attempt in range(6):
        try:
            aws("start-instances", "--instance-ids", INSTANCE)
            return
        except RuntimeError as e:
            if "IncorrectSpotRequestState" not in str(e) or attempt == 5:
                raise
            time.sleep(5)


def counts() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        data = json.loads(COUNTS.read_text())
    except (OSError, ValueError):
        data = {}
    if data.get("day") != today:
        data = {"day": today, "starts": {}}
    return data


def starts_left(user: str):
    if user == ADMIN:
        return None
    return max(0, LIMIT - counts()["starts"].get(user, 0))


def count_start(user: str) -> None:
    data = counts()
    data["starts"][user] = data["starts"].get(user, 0) + 1
    COUNTS.write_text(json.dumps(data))


class Handler(BaseHTTPRequestHandler):
    def user(self) -> str:
        return self.headers.get("X-User", "") or "nobody"

    def reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/start":
            return self.reply(404, {"error": "not found"})
        user = self.user()
        try:
            self.reply(200, {"user": user, "admin": user == ADMIN, "left": starts_left(user), "state": state()})
        except Exception as e:
            self.reply(500, {"error": str(e)})

    def do_POST(self):
        if self.path != "/start":
            return self.reply(404, {"error": "not found"})
        user = self.user()
        left = starts_left(user)
        if left == 0:
            return self.reply(429, {"error": f"you have used today's {LIMIT} starts; ask {ADMIN}, or try tomorrow",
                                    "left": 0, "state": None})
        try:
            current = state()
            if current == "stopped":
                start_instance()
                if left is not None:
                    count_start(user)
                    left -= 1
                print(f"{user} started {INSTANCE}", flush=True)
                self.reply(200, {"started": True, "state": "pending", "left": left})
            elif current in ("pending", "running"):
                self.reply(200, {"started": False, "state": current, "left": left})
            else:  # stopping: a start is refused by AWS until it is stopped
                self.reply(409, {"error": f"the machine is {current}; try again in a minute", "state": current, "left": left})
        except Exception as e:
            self.reply(500, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 9000), Handler).serve_forever()
