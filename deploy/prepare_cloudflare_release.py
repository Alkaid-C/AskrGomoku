#!/usr/bin/env python3
"""Build and validate the immutable Cloudflare Static Assets release."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import onnx  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web_app"
DIST_ROOT = ROOT / "cloudflare_dist"

BUILD_FORMAT_VERSION = "askr-cloudflare-release-v2"
WORKERS_ASSET_LIMIT = 25 * 1024 * 1024
ORT_VERSION = "1.27.0"
TERSER_VERSION = "5.19.2"

EXPECTED_VENDOR_FILES = {
    "ort.webgpu.min.js",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
}
EXPECTED_VENDOR_SHA256 = {
    "ort.webgpu.min.js": "a3f348c2fec54c8c4ac503967c33c1943a79e96dba40fb867ab0f501be94bf84",
    "ort-wasm-simd-threaded.asyncify.mjs": "7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655",
    "ort-wasm-simd-threaded.asyncify.wasm": "7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a",
}
EXPECTED_MODEL_FILES = {
    "dial.onnx",
    "cello.onnx",
    "curtain.onnx",
    "melody-eval-cache.bin",
}
EXPECTED_MODEL_SHA256 = {
    "dial.onnx": "79da078bb3b35e86046f7090747d0c16ba8136fc5847cbc0b16ed8745ded688d",
    "cello.onnx": "7fb0fb918590f30dcab60d45bcf66187cf948c3fb3c29d3d9e0b76183a34ce57",
    "curtain.onnx": "9da8ce795c37ccb74e26dab9816729c9a4fd40733a729571958810dd164f104d",
    "melody-eval-cache.bin": "96a72549a3d6f7d5d47dc7f69863988393a3f3120cda202ebdfa42ca74a8e433",
}

EVAL_CACHE_MAGIC = b"GMKECACH"
EVAL_CACHE_HEADER_SIZE = 64
EVAL_CACHE_RECORD_SIZE = 964
EVAL_CACHE_MODEL_HASH_OFFSET = 0x18


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_exact_files(directory: Path, expected: set[str]) -> None:
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Unexpected files in {directory}: missing={missing}, extra={extra}"
        )


def _validate_onnx(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    external = [
        tensor.name
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        or len(tensor.external_data) != 0
    ]
    if external:
        raise RuntimeError(
            f"{path.name} is not a single-file ONNX model; "
            f"external initializers: {external[:5]}"
        )

    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {"board_state"}:
        raise RuntimeError(f"{path.name} has unexpected inputs: {sorted(inputs)}")
    if set(outputs) != {"policy_logits", "value"}:
        raise RuntimeError(f"{path.name} has unexpected outputs: {sorted(outputs)}")

    input_shape = [
        dimension.dim_value
        for dimension in inputs["board_state"].type.tensor_type.shape.dim
    ]
    policy_shape = [
        dimension.dim_value
        for dimension in outputs["policy_logits"].type.tensor_type.shape.dim
    ]
    value_shape = [
        dimension.dim_value
        for dimension in outputs["value"].type.tensor_type.shape.dim
    ]
    if input_shape != [2, 15, 15] or policy_shape != [225] or value_shape != []:
        raise RuntimeError(
            f"{path.name} has unexpected I/O shapes: "
            f"input={input_shape}, policy={policy_shape}, value={value_shape}"
        )


def _validate_eval_cache(cache_path: Path, model_path: Path) -> None:
    data = cache_path.read_bytes()
    if len(data) < EVAL_CACHE_HEADER_SIZE or data[:8] != EVAL_CACHE_MAGIC:
        raise RuntimeError("Melody eval cache has an invalid or truncated header")

    version, board_size, policy_size, entry_count = struct.unpack_from(
        "<HHHH", data, 0x08
    )
    record_size, flags = struct.unpack_from("<II", data, 0x10)
    expected_length = EVAL_CACHE_HEADER_SIZE + entry_count * record_size
    if (
        version != 1
        or board_size != 15
        or policy_size != 225
        or record_size != EVAL_CACHE_RECORD_SIZE
        or flags != 0
        or expected_length != len(data)
        or data[0x38:0x40] != bytes(8)
    ):
        raise RuntimeError("Melody eval cache metadata is inconsistent")

    expected_hash = bytes.fromhex(_sha256_file(model_path))
    actual_hash = data[
        EVAL_CACHE_MODEL_HASH_OFFSET : EVAL_CACHE_MODEL_HASH_OFFSET + 32
    ]
    if actual_hash != expected_hash:
        raise RuntimeError("Melody eval cache does not match curtain.onnx")


def _source_asset_paths() -> list[Path]:
    paths = [WEB_ROOT / "index.html", WEB_ROOT / "styles.css"]
    paths.extend(sorted((WEB_ROOT / "js").glob("*.js")))
    paths.extend(WEB_ROOT / "vendor" / name for name in sorted(EXPECTED_VENDOR_FILES))
    paths.extend(WEB_ROOT / "models" / name for name in sorted(EXPECTED_MODEL_FILES))
    return paths


def _validate_sources(paths: list[Path]) -> None:
    _assert_exact_files(WEB_ROOT / "vendor", EXPECTED_VENDOR_FILES)
    _assert_exact_files(WEB_ROOT / "models", EXPECTED_MODEL_FILES)

    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Required release asset is missing: {path}")
        if path.stat().st_size > WORKERS_ASSET_LIMIT:
            raise RuntimeError(
                f"{path} is {path.stat().st_size} bytes, above the "
                f"Cloudflare Static Assets limit of {WORKERS_ASSET_LIMIT} bytes"
            )

    wasm = WEB_ROOT / "vendor" / "ort-wasm-simd-threaded.asyncify.wasm"
    if wasm.read_bytes()[:4] != b"\0asm":
        raise RuntimeError("The vendored ORT WASM file has invalid magic")

    try:
        terser_version = subprocess.run(
            ["terser", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"terser {TERSER_VERSION} is required to build production assets"
        ) from error
    if terser_version != f"terser {TERSER_VERSION}":
        raise RuntimeError(
            f"Expected terser {TERSER_VERSION}, found {terser_version!r}"
        )

    bundle = (WEB_ROOT / "vendor" / "ort.webgpu.min.js").read_bytes()
    fetch_script = (WEB_ROOT / "fetch_vendor.sh").read_text()
    if ORT_VERSION.encode() not in bundle or f"VERSION={ORT_VERSION}" not in fetch_script:
        raise RuntimeError(
            f"Vendored ORT files and fetch_vendor.sh must both pin {ORT_VERSION}"
        )
    for name, expected_hash in EXPECTED_VENDOR_SHA256.items():
        actual_hash = _sha256_file(WEB_ROOT / "vendor" / name)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Vendored {name} failed SHA-256 verification: {actual_hash}"
            )
    for name, expected_hash in EXPECTED_MODEL_SHA256.items():
        actual_hash = _sha256_file(WEB_ROOT / "models" / name)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Release model {name} failed SHA-256 verification: {actual_hash}"
            )

    for name in ("dial.onnx", "cello.onnx", "curtain.onnx"):
        _validate_onnx(WEB_ROOT / "models" / name)
    _validate_eval_cache(
        WEB_ROOT / "models" / "melody-eval-cache.bin",
        WEB_ROOT / "models" / "curtain.onnx",
    )


def _release_id(paths: list[Path]) -> str:
    digest = hashlib.sha256(
        f"{BUILD_FORMAT_VERSION}\0terser-{TERSER_VERSION}".encode()
    )
    for path in paths:
        logical_path = path.relative_to(WEB_ROOT).as_posix()
        digest.update(b"\0")
        digest.update(logical_path.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()[:20]


def _build_index(release_id: str) -> str:
    asset_base = f"assets/{release_id}/"
    html = (WEB_ROOT / "index.html").read_text()

    html, meta_count = re.subn(
        r'(<meta name="gomoku-asset-base" content=")[^"]*(">)',
        rf"\g<1>{asset_base}\g<2>",
        html,
        count=1,
    )
    if meta_count != 1:
        raise RuntimeError("Could not set gomoku-asset-base in index.html")

    html, style_count = re.subn(
        r'href="styles\.css(?:\?[^"]*)?"',
        f'href="{asset_base}styles.css"',
        html,
        count=1,
    )
    if style_count != 1:
        raise RuntimeError("Could not rewrite styles.css in index.html")

    html, vendor_count = re.subn(
        r'src="vendor/([^"?]+)(?:\?[^"]*)?"',
        rf'src="{asset_base}vendor/\1"',
        html,
    )
    html, js_count = re.subn(
        r'src="js/([^"?]+)(?:\?[^"]*)?"',
        rf'src="{asset_base}js/\1"',
        html,
    )
    if vendor_count != 1 or js_count == 0:
        raise RuntimeError(
            "Unexpected local script references in index.html: "
            f"vendor={vendor_count}, js={js_count}"
        )
    # There are no inline scripts or conditional comments in this document.
    # Strip authoring comments while preserving text-node whitespace.
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _security_headers(release_id: str) -> str:
    common = """/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), geolocation=(), microphone=(), payment=(), usb=()
  Content-Security-Policy-Report-Only: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'none'

/
  Cache-Control: public, max-age=0, must-revalidate

/index.html
  Cache-Control: public, max-age=0, must-revalidate

/release-manifest.json
  Cache-Control: public, max-age=0, must-revalidate

/assets/*
  Cache-Control: public, max-age=31536000, immutable
"""
    model_rules = []
    for name in ("dial.onnx", "cello.onnx", "curtain.onnx"):
        size = (WEB_ROOT / "models" / name).stat().st_size
        model_rules.append(
            f"\n/assets/{release_id}/models/{name}\n"
            "  Content-Type: application/x-protobuf\n"
            f"  X-Uncompressed-Length: {size}\n"
        )
    return common + "".join(model_rules)


def _strip_css_comments(css: str) -> str:
    """Remove CSS comments without touching comment-like text in strings."""
    output = []
    index = 0
    quote = None
    while index < len(css):
        char = css[index]
        if quote is not None:
            output.append(char)
            if char == "\\" and index + 1 < len(css):
                index += 1
                output.append(css[index])
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(char)
            index += 1
            continue
        if css.startswith("/*", index):
            end = css.find("*/", index + 2)
            if end == -1:
                raise RuntimeError("Unterminated CSS comment")
            index = end + 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _minify_javascript(source: Path, destination: Path) -> None:
    command = [
        "terser",
        str(source),
        "--ecma",
        "2020",
        "--compress",
        'passes=2,drop_console=["log","debug"]',
        "--define",
        "globalThis.GOMOKU_DEBUG=false",
        "--format",
        "comments=false",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to minify {source.name}: {error.stderr.strip()}"
        ) from error
    destination.write_text(result.stdout)


def _copy_release_assets(staging: Path, release_id: str) -> Path:
    release_root = staging / "assets" / release_id
    (release_root / "js").mkdir(parents=True)

    css = (WEB_ROOT / "styles.css").read_text()
    (release_root / "styles.css").write_text(_strip_css_comments(css))
    for path in sorted((WEB_ROOT / "js").glob("*.js")):
        _minify_javascript(path, release_root / "js" / path.name)
    shutil.copytree(WEB_ROOT / "vendor", release_root / "vendor")
    shutil.copytree(WEB_ROOT / "models", release_root / "models")
    return release_root


def _verify_built_index(staging: Path) -> None:
    html = (staging / "index.html").read_text()
    references = re.findall(r'(?:href|src)="([^"]+)"', html)
    for reference in references:
        if reference.startswith(("http://", "https://", "#")):
            continue
        local_path = reference.split("?", 1)[0]
        if not (staging / local_path).is_file():
            raise RuntimeError(f"Built index references a missing asset: {reference}")

    forbidden = {"bench.html", "test.html", "fetch_vendor.sh"}
    deployed_names = {path.name for path in staging.rglob("*") if path.is_file()}
    leaked = sorted(forbidden & deployed_names)
    if leaked:
        raise RuntimeError(f"Development files leaked into the release: {leaked}")

    for path in (staging / "assets").glob("*/js/*.js"):
        javascript = path.read_text()
        if "console.log(" in javascript or "console.debug(" in javascript:
            raise RuntimeError(f"Debug logging survived production minification: {path}")


def _write_manifest(staging: Path, release_id: str) -> None:
    files = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "release-manifest.json":
            relative = path.relative_to(staging).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    manifest = {
        "format": BUILD_FORMAT_VERSION,
        "release_id": release_id,
        "onnxruntime_web": ORT_VERSION,
        "production_transform": {
            "comments": "removed from app HTML/CSS/JS",
            "debug_console": ["log", "debug"],
            "terser": TERSER_VERSION,
        },
        "files": files,
    }
    (staging / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def build_release() -> str:
    paths = _source_asset_paths()
    _validate_sources(paths)
    release_id = _release_id(paths)

    staging = Path(tempfile.mkdtemp(prefix=".cloudflare-dist-", dir=ROOT))
    try:
        _copy_release_assets(staging, release_id)
        (staging / "index.html").write_text(_build_index(release_id))
        (staging / "_headers").write_text(_security_headers(release_id))
        _verify_built_index(staging)
        _write_manifest(staging, release_id)

        if DIST_ROOT.is_symlink():
            raise RuntimeError(f"Refusing to replace symlinked output: {DIST_ROOT}")
        if DIST_ROOT.exists():
            shutil.rmtree(DIST_ROOT)
        staging.rename(DIST_ROOT)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    total_size = sum(
        path.stat().st_size for path in DIST_ROOT.rglob("*") if path.is_file()
    )
    print(f"Cloudflare release ready: {DIST_ROOT}")
    print(f"Release ID: {release_id}")
    print(f"Files: {sum(1 for path in DIST_ROOT.rglob('*') if path.is_file())}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MiB")
    return release_id


if __name__ == "__main__":
    build_release()
