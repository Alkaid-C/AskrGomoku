# ONNX Model Deployment Guide

Integration reference for the exported Gomoku ONNX models (as produced by
`export_onnx.py`). The in-repo consumer is `web_app/` — see `deploy/CLAUDE.md`
for the app architecture; this file documents only the model contract.

**Format**: ONNX Opset 21, single self-contained file (no `.onnx.data` sidecar)
**Batch size**: fixed at 1 (the wrapper adds the batch dimension internally)

---

## Model Input

### `board_state`
- **Shape**: `[2, 15, 15]`, **Type**: `float32` (no batch dimension)
- **Channel 0**: current player's stones (1.0 = stone, 0.0 = empty)
- **Channel 1**: opponent's stones

The board is always encoded from the **side-to-move's perspective** — the
channels swap every ply. The constant all-ones board-mask channel the network
was trained with is added inside the exported graph; callers provide only the
two stone planes.

```javascript
const boardState = new Float32Array(2 * 15 * 15);
// index = channel * 225 + row * 15 + col   (row-major)
boardState[0 * 225 + 7 * 15 + 7] = 1.0;  // current player at (7,7)
boardState[1 * 225 + 7 * 15 + 8] = 1.0;  // opponent at (7,8)
const tensor = new ort.Tensor('float32', boardState, [2, 15, 15]);
```

## Model Outputs

### `policy_logits`
- **Shape**: `[225]`, **Type**: `float32`
- **Raw logits** — no softmax, no temperature, no legality masking in the
  graph. Flat index = `row * 15 + col`.
- The caller must mask illegal squares and apply its own softmax (and
  temperature, if sampling). Doing this outside the graph avoids precision
  issues from softmaxing over illegal positions and lets one exported file
  serve every temperature. See `web_app/js/onnx-ai-player.js`
  (`_maskedSoftmax`) for the reference implementation.

### `value`
- **Shape**: scalar, **Type**: `float32`, range [−1, +1] (tanh-bounded)
- Position evaluation **from the current player's perspective**.
- For MCTS-trained checkpoints (stage 2), the value head is trained against
  `root_Q` — a visit-weighted soft average — not a minimax value. Average-style
  search backups (MCTS) match these semantics; hard max/negamax over noisy
  estimates amplifies noise.

## Inference (onnxruntime-web)

```javascript
// wasmPaths must be an ABSOLUTE URL (Firefox module resolution);
// numThreads is read once at the first wasm init and cannot change after.
ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency) : 1;

const session = await ort.InferenceSession.create('models/model.onnx', {
    executionProviders: ['wasm'],   // or ['webgpu']
    graphOptimizationLevel: 'all',
});
const out = await session.run({ board_state: tensor });
const logits = out.policy_logits.data;  // Float32Array[225]
const value = out.value.data[0];
```

Execution-provider notes (measured 2026-07, ORT 1.27):

- The fastest EP varies wildly by device: desktop Chrome WASM-4T ≈ WebGPU
  (~17–23 ms/inference), desktop Firefox WebGPU is ~10× slower than WASM,
  iOS Safari WebGPU is ~2.4× faster than WASM (threads do nothing there).
  Do not hardcode an EP; probe at load time (`web_app/js/ep-probe.js`).
- Multithreaded WASM requires `crossOriginIsolated`, i.e. the page must be
  served with `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. More than 4 threads never
  helped. Under COEP, cross-origin resources need cooperating CORS/CORP
  headers — hence ORT is vendored same-origin rather than loaded from a CDN.
- WebGPU's first runs include shader compilation (seconds; observed 13 s on
  a throttled laptop) — warm up before timing or playing.
- `session.run()` is not abortable; to bound a slow probe, race the promise
  against a timeout and abandon the orphaned run.

## Export Command Reference

```bash
cd deploy && python3 export_onnx.py --input checkpoint.pt --output model.onnx
```

The exporter loads `checkpoint['model_state_dict']` into `main/model.py`'s
`GomokuPolicyNet`, **decomposes every `nn.GroupNorm` into primitive ops**
(onnxruntime-web's WebGPU EP has no `GroupNormalization` kernel; undecomposed
graphs shatter into GPU/CPU partitions and run ~25× slower on WebGPU — do not
remove this step), wraps it for the interface above, re-saves as a single
file, and verifies ONNX-vs-PyTorch outputs to < 1e-5.
