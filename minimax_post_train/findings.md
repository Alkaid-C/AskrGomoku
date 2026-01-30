# Review Findings

## Stage 0 — Spec Mapping (POST_TRAIN_SPEC.md ↔ IMPLEMENTATION_NOTES.md)
- Candidate count mismatch: spec says 6 candidates at every node; implementation uses 6 at root and 5 internally (`gomoku.py`).
- Ranking-inside scope mismatch: spec constrains c1..c4; implementation also includes c4 vs c5 (top-5) in ranking loss (`training.py`).
- Sample schema mismatch: spec records sorted_candidates [c1..c4]; implementation stores top-5 + Q values (`gomoku.py`).
- Search cache reuse not implemented: spec mentions subtree reuse by move; no cache present (`gomoku.py`).
- Resume state mismatch: notes call out tracking unfrozen blocks in `training_state.json`, but save/load do not include it (`main.py`).
- Spec mentions branch factor k=3; implementation uses 6/5 candidates, so intent needs clarification (`gomoku.py`).

## Stage 1 — Search Data Generation (Candidates + Negamax)
- Internal candidate count deviates from spec: internal nodes use 5 candidates (top-4 + random-1) vs spec 6 everywhere (`gomoku.py`).
- Batched candidate generation may include illegal moves when legal count < top_k: `generate_candidates_batched` always takes `top_k` even if fewer legal moves exist, because masked logits are still ranked (`gomoku.py`).
- Leaf value sign handling only correct for odd depths: parity-based negation before backup makes even depths inconsistent with negamax semantics (depth=3 works; depth=2/4 would not) (`gomoku.py`).
- Opponent move search uses opponent model for leaf evaluation: spec states leaf eval always uses current (training) model; `play_episodes_with_search` passes opponent as current_model for opponent turns (`gomoku.py`).
- Candidate list length not guaranteed when board has very few legal moves: fallback returns fewer than required instead of enforcing fixed-size unique candidate set (`gomoku.py`).
- Search cache reuse not implemented despite spec mention (no subtree reuse keyed by last move) (`gomoku.py`).
