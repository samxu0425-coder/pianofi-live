#!/bin/bash
# Runs every minute on the GPU box (systemd timer, installed by deploy/run.sh). When no engine
# has run for IDLE_MINUTES in a row the machine shuts down, which for this spot instance means
# "stopped": the disk, the containers and the address survive, and the front's offline page
# offers a Start button. A minute where the service does not answer counts as idle too (nobody
# can play then either); the first build takes a few minutes, well under the default 30.
IDLE_MINUTES=${IDLE_MINUTES:-30}
COUNT=/run/pianofi-idle-minutes
running=$(curl -fsS --max-time 5 http://127.0.0.1:8000/status 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["running"])' 2>/dev/null || echo 0)
if [ "$running" != "0" ]; then
  echo 0 > "$COUNT"
  exit 0
fi
idle=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$idle" > "$COUNT"
if [ "$idle" -ge "$IDLE_MINUTES" ]; then
  echo "pianofi: no engine for $idle minutes, stopping the machine"
  shutdown -h now
fi
