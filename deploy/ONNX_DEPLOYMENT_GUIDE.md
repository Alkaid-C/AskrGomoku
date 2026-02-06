# ONNX Model Deployment Guide

**Model**: Gomoku Policy Network (3.55M parameters)
**Format**: ONNX Opset 21
**Target Runtime**: onnxruntime-web (WASM backend)
**File Size**: ~0.3 MB (compressed), ~14 MB (in-memory)
**Batch Size**: Single inference only (no batch dimension)

---

## Model Input

### `board_state`
- **Shape**: `[2, 15, 15]`
- **Type**: `float32`
- **Description**: 15×15 Gomoku board state from current player's perspective (no batch dimension)

#### Channel Layout
- **Channel 0**: Current player's stones (1.0 = stone present, 0.0 = empty)
- **Channel 1**: Opponent's stones (1.0 = stone present, 0.0 = empty)

**Note**: The model internally adds a constant mask channel (all 1.0s) as the 3rd channel before inference. You only need to provide 2 channels.

#### Example Input Construction (JavaScript)
```javascript
// Create empty board state (2 channels, no batch dimension)
const boardState = new Float32Array(2 * 15 * 15);

// Helper to set a position
function setStone(channel, row, col, value) {
  const index = channel * (15 * 15) + row * 15 + col;
  boardState[index] = value;
}

// Example: Current player at (7, 7), opponent at (7, 8)
setStone(0, 7, 7, 1.0);  // Current player (channel 0)
setStone(1, 7, 8, 1.0);  // Opponent (channel 1)

// No need to set board mask - it's added automatically inside the model!
```

---

## Model Outputs

### Output 1: `policy_probs`
- **Shape**: `[15, 15]`
- **Type**: `float32`
- **Description**: Probability distribution over all board positions (2D grid, no batch dimension)
- **Properties**:
  - Sum equals 1.0 (valid probability distribution)
  - Direct board layout: `policy_probs[row][col]` = probability for position (row, col)
  - Higher values indicate more favorable moves

#### Accessing Policy for a Position
```javascript
// Get probability for position (row, col)
// ONNX returns Float32Array[225] in row-major order
const prob = policyProbs.data[row * 15 + col];
```

### Output 2: `value`
- **Shape**: scalar (no dimensions)
- **Type**: `float32`
- **Description**: Current player's win probability estimation
- **Range**: [-1.0, +1.0]
  - **+1.0**: Current player is winning
  - **0.0**: Even position
  - **-1.0**: Current player is losing

---

## Inference Example (onnxruntime-web)

### Installation
```bash
npm install onnxruntime-web
```

### JavaScript Example
```javascript
import * as ort from 'onnxruntime-web';

// Set backend to WASM (required for GroupNormalization support)
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

// Load model
const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: ['wasm']
});

// Prepare input (2 channels, no batch dimension)
const boardState = new Float32Array(2 * 15 * 15);
// ... fill boardState as shown above ...

const inputTensor = new ort.Tensor('float32', boardState, [2, 15, 15]);

// Run inference
const outputs = await session.run({ board_state: inputTensor });

// Extract outputs
const policyProbs = outputs.policy_probs.data;  // Float32Array[225] (flattened from [15, 15])
const value = outputs.value.data;               // scalar float

// Find best move (policy is in row-major order: index = row * 15 + col)
let bestRow = 0;
let bestCol = 0;
let bestProb = -1;

for (let row = 0; row < 15; row++) {
  for (let col = 0; col < 15; col++) {
    const index = row * 15 + col;
    if (policyProbs[index] > bestProb) {
      bestProb = policyProbs[index];
      bestRow = row;
      bestCol = col;
    }
  }
}

console.log(`Best move: (${bestRow}, ${bestCol}), probability: ${bestProb.toFixed(4)}`);
console.log(`Position evaluation: ${value.toFixed(4)}`);
```

---

## Important Notes

### Temperature
- **Softmax temperature is baked into the model** during export
- Cannot be changed at runtime without re-exporting
- Different models can be exported with different temperatures:
  - Low temperature (e.g., 0.1): Sharper, more deterministic play
  - High temperature (e.g., 1.0): More exploratory, diverse play

### Sampling Moves
To sample a move according to the policy distribution:
```javascript
function sampleMove(policyProbs) {
  // policyProbs is Float32Array[225] in row-major order
  const rand = Math.random();
  let cumulative = 0;

  for (let i = 0; i < 225; i++) {
    cumulative += policyProbs[i];
    if (rand < cumulative) {
      return {
        row: Math.floor(i / 15),
        col: i % 15
      };
    }
  }

  return { row: 14, col: 14 };  // Fallback
}
```

### Legal Move Masking
The model outputs probabilities for all 225 positions. You should:
1. Mask out illegal moves (already occupied positions)
2. Renormalize probabilities
```javascript
function maskIllegalMoves(policyProbs, boardState) {
  // policyProbs: Float32Array[225] in row-major order
  // boardState: 2D array or check function to determine if position is occupied

  const masked = new Float32Array(225);
  let sum = 0;

  for (let row = 0; row < 15; row++) {
    for (let col = 0; col < 15; col++) {
      const index = row * 15 + col;
      const isLegal = !isOccupied(row, col, boardState);  // Your logic here

      masked[index] = isLegal ? policyProbs[index] : 0;
      sum += masked[index];
    }
  }

  // Renormalize
  if (sum > 0) {
    for (let i = 0; i < 225; i++) {
      masked[i] /= sum;
    }
  }

  return masked;
}

function isOccupied(row, col, boardState) {
  // Check if position (row, col) has a stone
  // boardState could be your game state representation
  return boardState[row][col] !== 0;
}
```

---

## Performance Tips

1. **Reuse session**: Create the `InferenceSession` once, reuse for all inferences
2. **WebAssembly SIMD**: Enable for ~2x speedup (already shown in example)
3. **Multi-threading**: Adjust `ort.env.wasm.numThreads` based on target device
4. **Batch size**: Currently fixed at 1; if you need to evaluate multiple positions, re-export with dynamic batch

---

## Export Command Reference

To export a model with different temperature:
```bash
python3 export_onnx.py --input checkpoint.pt --output model.onnx --temp 0.5
```

---

## Model Architecture Summary

- **Input**: Board state (current player perspective)
- **Architecture**: ResNet-style with GroupNorm, 18 residual blocks, 96 channels
- **Policy Head**: Predicts move probabilities (with temperature applied)
- **Value Head**: Predicts win probability

For implementation details, see `model.py` and `export_onnx.py`.
