#!/bin/bash
# First boot of the front machine. This is a Terraform template: the dollar-brace fields are filled in by front.tf.
# A t4g.nano has 512 MB, too little for Docker, so Caddy and the start handler run as plain processes under systemd.
set -euo pipefail
mkdir -p /opt/pianofi/front
curl -fsSL -o /tmp/caddy.tar.gz "https://github.com/caddyserver/caddy/releases/download/v${caddy_version}/caddy_${caddy_version}_linux_arm64.tar.gz"
tar -xzf /tmp/caddy.tar.gz -C /usr/local/bin caddy
chmod 755 /usr/local/bin/caddy
# What both services read (deploy/front/install.sh points them at it). Root only: it holds the password hashes.
cat > /opt/pianofi/env <<'ENV'
SITE=${site}
ENGINE=${engine_ip}
INSTANCE=${instance_id}
AWS_DEFAULT_REGION=${region}
AUTH_USER=${auth_user}
AUTH_HASH=${auth_hash}
GUEST_USER=${guest_user}
GUEST_HASH=${guest_hash}
GUEST_STARTS_PER_DAY=${guest_starts}
ENV
chmod 600 /opt/pianofi/env
cat > /opt/pianofi/front/Caddyfile <<'CADDY'
${caddyfile}
CADDY
cat > /opt/pianofi/front/offline.html <<'PAGE'
${offline}
PAGE
cat > /opt/pianofi/front/start.py <<'PY'
${start_py}
PY
cat > /opt/pianofi/front/install.sh <<'SH'
${install_sh}
SH
bash /opt/pianofi/front/install.sh
