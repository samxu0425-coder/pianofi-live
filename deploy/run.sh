#!/bin/bash
# Start (or restart) the PianoFi Live service on the GPU machine.
#   deploy/run.sh                        service only: ports 8000/8001 for the front machine (RUN.md step 11)
#   deploy/run.sh SITE AUTH_USER AUTH_HASH   single-box setup: also Caddy with HTTPS and a password
# Run from the folder that holds Dockerfile, ui/, models/ and songs/. Builds the service
# image first (a few minutes the first time, seconds after). AUTH_HASH is a bcrypt hash from
#   docker run --rm caddy:2 caddy hash-password --plaintext '...'
# Environment: ADMIN (default sam) is the login that may add songs; IDLE_MINUTES (default 30)
# is how long the box runs with no engine before it stops itself (0 = never).
set -euo pipefail
cd "$(dirname "$0")/.."
ADMIN=${ADMIN:-sam}
IDLE_MINUTES=${IDLE_MINUTES:-30}

sudo docker build -q --target service -t pianofi-live-service . >/dev/null
sudo docker network create pianofi 2>/dev/null || true
sudo docker rm -f service caddy 2>/dev/null || true

# Stop the machine when nobody has played for a while (deploy/idle_stop.sh, once a minute).
if [ "$IDLE_MINUTES" -gt 0 ]; then
  sudo install -m 755 deploy/idle_stop.sh /usr/local/bin/pianofi-idle-stop
  sudo tee /etc/systemd/system/pianofi-idle-stop.service >/dev/null <<UNIT
[Unit]
Description=PianoFi Live: stop the machine after $IDLE_MINUTES idle minutes
[Service]
Type=oneshot
Environment=IDLE_MINUTES=$IDLE_MINUTES
ExecStart=/usr/local/bin/pianofi-idle-stop
UNIT
  sudo tee /etc/systemd/system/pianofi-idle-stop.timer >/dev/null <<'UNIT'
[Unit]
Description=PianoFi Live: idle check every minute
[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now pianofi-idle-stop.timer
  sudo systemctl restart pianofi-idle-stop.timer
else
  sudo systemctl disable --now pianofi-idle-stop.timer 2>/dev/null || true
fi

# Started after the systemd setup above on purpose: a `systemctl daemon-reload` takes the GPU away
# from containers that are already running (nvidia-smi inside says "Failed to initialize NVML"),
# and the engines then fall back to the CPU.
# TensorRT builds its engines the first time they are used (44 s on the T4) and keeps them in
# models/trt_cache. The service mounts models read-only, so build them here, with a 5 s run.
song=$(ls songs/*.wav 2>/dev/null | head -1)
sudo docker run --rm --gpus all -v "$PWD/models:/work/models" -v "$PWD/songs:/work/songs:ro" \
  pianofi-live-service /app/pianofi_live --wav "$song" --stride 16 --delay 16 --trt --mixed --fast \
  --max-seconds 5 --out /tmp/warm 2>&1 | grep -E "provider|unavailable" || echo "warning: the TensorRT warm-up run failed"

# One engine per page, five at once on TensorRT with the mixed-precision graphs: what fits in a
# T4's memory (results.md, "Tesla T4 with TensorRT"; four with --fp16, two in FP32). The ports
# are only reachable from the front machine (security group), or from Caddy on this box.
sudo docker run -d --name service --network pianofi --restart unless-stopped --gpus all \
  -p 8000:8000 -p 8001:8001 \
  -v "$PWD/models:/work/models:ro" -v "$PWD/songs:/work/songs" -v "$PWD/runs:/work/runs" \
  pianofi-live-service python3 /app/ui/server.py --host 0.0.0.0 --songs /work/songs --max-sessions 5 --trt --admin "$ADMIN"

if [ $# -eq 3 ]; then
  site=$1; user=$2; hash=$3
  sudo docker run -d --name caddy --network pianofi --restart unless-stopped \
    -p 80:80 -p 443:443 -v "$PWD/deploy/Caddyfile:/etc/caddy/Caddyfile:ro" -v caddy_data:/data \
    -e SITE="$site" -e AUTH_USER="$user" -e AUTH_HASH="$hash" caddy:2
  echo "up: https://$site (user $user)"
else
  echo "service up on ports 8000 and 8001; idle stop after $IDLE_MINUTES minutes"
fi
