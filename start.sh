#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ ! -f .env ]]; then
  printf 'Copy %s/.env.example to .env and set BIND_HOST.\n' "$HERE" >&2
  exit 1
fi

compose=(docker compose --env-file .env -f compose.yaml)
port="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["embedding"]["environment"]["EMBEDDING_PORT"])')"
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
  printf 'Invalid PORT: %s\n' "$port" >&2
  exit 1
fi
if [[ -z "$("${compose[@]}" ps --status running -q embedding)" && -n "$(ss -H -ltn "sport = :$port")" ]]; then
  printf 'Port %s is already in use. Set a free PORT in .env.\n' "$port" >&2
  exit 1
fi

exec "${compose[@]}" up -d
