#!/usr/bin/env python3
"""Verify the production Cloudflare deployment without downloading models."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.client import HTTPMessage

EXPECTED_MODEL_SIZES = {
    "dial.onnx": 13_876_938,
    "cello.onnx": 15_605_004,
    "curtain.onnx": 15_604_518,
}


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
) -> tuple[int, HTTPMessage, bytes]:
    request_headers = {"User-Agent": "askr-gomoku-deploy-verifier/1"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, method=method, headers=request_headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.headers, response.read()


def _require_header(headers: HTTPMessage, name: str, expected: str) -> None:
    actual = headers.get(name)
    if actual != expected:
        raise RuntimeError(f"Expected {name}: {expected!r}, got {actual!r}")


def _require_cache_policy(headers: HTTPMessage, immutable: bool) -> None:
    value = (headers.get("Cache-Control") or "").lower()
    required = (
        ("max-age=31536000", "immutable")
        if immutable
        else ("max-age=0", "must-revalidate")
    )
    missing = [directive for directive in required if directive not in value]
    if missing:
        raise RuntimeError(
            f"Cache-Control {value!r} is missing directives: {missing}"
        )


def _expect_not_found(url: str) -> None:
    try:
        _request(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return
        raise RuntimeError(f"{url} returned HTTP {error.code}, expected 404") from error
    raise RuntimeError(f"Development-only asset is publicly reachable: {url}")


def verify(base_url: str) -> None:
    base_url = base_url.rstrip("/") + "/"
    asset_base = ""
    release_id = ""
    for attempt in range(6):
        status, index_headers, index_body = _request(base_url)
        if status != 200:
            raise RuntimeError(f"Root returned HTTP {status}")

        _require_header(index_headers, "Cross-Origin-Opener-Policy", "same-origin")
        _require_header(index_headers, "Cross-Origin-Embedder-Policy", "require-corp")
        _require_header(index_headers, "Cross-Origin-Resource-Policy", "same-origin")
        _require_header(index_headers, "X-Content-Type-Options", "nosniff")
        _require_header(index_headers, "X-Frame-Options", "DENY")
        _require_cache_policy(index_headers, immutable=False)

        index_html = index_body.decode("utf-8")
        match = re.search(
            r'<meta name="gomoku-asset-base" content="(assets/([0-9a-f]{20})/)">',
            index_html,
        )
        if not match:
            raise RuntimeError("Production index has no content-addressed asset base")
        asset_base, release_id = match.groups()

        manifest_url = base_url + "release-manifest.json"
        _, manifest_headers, manifest_body = _request(manifest_url)
        _require_cache_policy(manifest_headers, immutable=False)
        manifest = json.loads(manifest_body)
        manifest_release_id = manifest.get("release_id")
        if manifest_release_id == release_id:
            break
        if attempt == 5:
            raise RuntimeError(
                "Production index and release manifest IDs do not match: "
                f"{release_id} != {manifest_release_id}"
            )
        time.sleep(2)

    immutable_paths = [
        asset_base + "styles.css",
        asset_base + "js/game-controller.js",
        asset_base + "vendor/ort-wasm-simd-threaded.asyncify.wasm",
    ]
    for path in immutable_paths:
        status, headers, _ = _request(base_url + path, method="HEAD")
        if status != 200:
            raise RuntimeError(f"{path} returned HTTP {status}")
        _require_cache_policy(headers, immutable=True)
        _require_header(headers, "Cross-Origin-Resource-Policy", "same-origin")

    wasm_url = base_url + asset_base + "vendor/ort-wasm-simd-threaded.asyncify.wasm"
    _, wasm_headers, _ = _request(wasm_url, method="HEAD")
    content_type = (wasm_headers.get("Content-Type") or "").split(";", 1)[0]
    if content_type != "application/wasm":
        raise RuntimeError(f"Unexpected WASM Content-Type: {content_type!r}")

    for name, expected_size in EXPECTED_MODEL_SIZES.items():
        model_url = base_url + asset_base + "models/" + name
        _, headers, _ = _request(
            model_url,
            method="HEAD",
            headers={"Accept-Encoding": "gzip"},
        )
        _require_cache_policy(headers, immutable=True)
        content_type = (headers.get("Content-Type") or "").split(";", 1)[0]
        if content_type != "application/x-protobuf":
            raise RuntimeError(f"Unexpected {name} Content-Type: {content_type!r}")
        decoded_size = int(headers.get("X-Uncompressed-Length") or 0)
        if decoded_size != expected_size:
            raise RuntimeError(
                f"Unexpected {name} decoded size: {decoded_size}, "
                f"expected {expected_size}"
            )
        encoding = headers.get("Content-Encoding")
        if encoding:
            print(f"{name}: Cloudflare compression active ({encoding})")
        else:
            print(
                f"{name}: warning — HEAD did not expose Content-Encoding; "
                "confirm compression with browser DevTools"
            )

    for name in ("test.html", "bench.html", "fetch_vendor.sh"):
        _expect_not_found(base_url + name)

    print(f"Cloudflare deployment verified: {base_url}")
    print(f"Release ID: {release_id}")
    print("Isolation, cache, MIME, model-size, and production-exclusion checks passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url",
        nargs="?",
        default="https://gomoku.sance.xyz/",
        help="Deployment base URL",
    )
    args = parser.parse_args()
    verify(args.url)


if __name__ == "__main__":
    main()
