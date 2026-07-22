#!/usr/bin/env bash
# Rebuild vendor/ (git-ignored onnxruntime-web binaries) from the npm registry.
# Why ORT is self-hosted, and which files are kept: see deploy/CLAUDE.md.
set -euo pipefail
cd "$(dirname "$0")"

VERSION=1.27.0
FILES=(
    ort.webgpu.min.js
    ort-wasm-simd-threaded.asyncify.mjs
    ort-wasm-simd-threaded.asyncify.wasm
)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${VERSION}.tgz" \
    -o "$tmp/pkg.tgz"
tar -xzf "$tmp/pkg.tgz" -C "$tmp"

mkdir -p vendor
for f in "${FILES[@]}"; do
    cp "$tmp/package/dist/$f" vendor/
done

echo "vendor/ populated with onnxruntime-web ${VERSION}:"
ls -lh vendor/
