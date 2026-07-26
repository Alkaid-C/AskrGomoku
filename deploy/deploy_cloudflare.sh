#!/usr/bin/env bash
# Build, validate, and deploy the assets-only Cloudflare Worker.
set -euo pipefail
cd "$(dirname "$0")"

: "${CLOUDFLARE_API_TOKEN:?Set CLOUDFLARE_API_TOKEN before deploying}"

WRANGLER_VERSION=4.114.0
export npm_config_cache="${TMPDIR:-/tmp}/askr-gomoku-npm-cache"
export XDG_CONFIG_HOME="${TMPDIR:-/tmp}/askr-gomoku-config"
export WRANGLER_SEND_METRICS=false

python3 prepare_cloudflare_release.py
npx --yes "wrangler@${WRANGLER_VERSION}" deploy --config wrangler.jsonc
python3 verify_cloudflare_deployment.py https://gomoku.sance.xyz/
