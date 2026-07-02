#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p certs

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "${IP:-}" ]; then
  IP="127.0.0.1"
fi

openssl req -x509 -newkey rsa:2048 \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 \
  -nodes \
  -subj "/CN=foxy.local" \
  -addext "subjectAltName=DNS:foxy.local,DNS:localhost,IP:127.0.0.1,IP:10.42.0.1,IP:${IP}"

echo "Created certs/cert.pem and certs/key.pem"
echo "Detected IP: ${IP}"
