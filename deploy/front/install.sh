#!/bin/bash
# Install (or update) the front machine's two services from the files next to this script:
#   caddy          HTTPS, the two logins, the proxy to the GPU box, the offline page
#   pianofi-start  the "Start the engine" button's handler (start.py), behind Caddy on 127.0.0.1:9000
# Both read /opt/pianofi/env (written on first boot by infra/deploy/front_user_data.sh).
# Run as root. Safe to run again: it copies the files and restarts the services.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /opt/pianofi/srv /opt/pianofi/data
install -m 644 Caddyfile /opt/pianofi/Caddyfile
install -m 644 offline.html /opt/pianofi/srv/offline.html
install -m 755 start.py /opt/pianofi/start.py

cat > /etc/systemd/system/caddy.service <<'UNIT'
[Unit]
Description=Caddy for PianoFi Live
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/opt/pianofi/env
Environment=XDG_DATA_HOME=/opt/pianofi/data
Environment=XDG_CONFIG_HOME=/opt/pianofi/data
ExecStart=/usr/local/bin/caddy run --config /opt/pianofi/Caddyfile
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/pianofi-start.service <<'UNIT'
[Unit]
Description=PianoFi Live: start the GPU box on request
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/opt/pianofi/env
ExecStart=/usr/bin/python3 /opt/pianofi/start.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now caddy pianofi-start
systemctl restart caddy pianofi-start
echo "front services installed"
