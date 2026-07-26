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
declare -A SHA256=(
    [ort.webgpu.min.js]=a3f348c2fec54c8c4ac503967c33c1943a79e96dba40fb867ab0f501be94bf84
    [ort-wasm-simd-threaded.asyncify.mjs]=7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655
    [ort-wasm-simd-threaded.asyncify.wasm]=7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a
)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${VERSION}.tgz" \
    -o "$tmp/pkg.tgz"
tar -xzf "$tmp/pkg.tgz" -C "$tmp"

for f in "${FILES[@]}"; do
    actual=$(sha256sum "$tmp/package/dist/$f" | cut -d' ' -f1)
    if [[ "$actual" != "${SHA256[$f]}" ]]; then
        echo "SHA-256 mismatch for onnxruntime-web ${VERSION} artifact: $f" >&2
        exit 1
    fi
done

mkdir -p vendor
for f in "${FILES[@]}"; do
    cp "$tmp/package/dist/$f" vendor/
done

echo "vendor/ populated with onnxruntime-web ${VERSION}:"
ls -lh vendor/
