# Review Findings (Synthesized)

## 1) **Clear bug** (sorted by severity)
1. **Resume boundary can skip training newly-unfrozen params**; Location: `main.py:347-372`, `training.py:859-873`; Does: on resume, applies freeze schedule for `start_update+1` and keeps loaded optimizer if `unfrozen_blocks` matches, even if optimizer param groups don't include newly-unfrozen params; Should: force optimizer recreation on resume (or verify param coverage) when schedule boundary is crossed; Consequence: newly-unfrozen layers may never get updated until the next unfreeze event.

## 2) **Risk**
- **Spec mismatch: branching factor**; `gomoku.py:40-55`; code uses 6 candidates per node while spec table states k=3; risk of training/search behavior diverging from intended spec and performance expectations.
- **Ranking loss scope stricter than spec**; `training.py:913-920`; includes (c4,c5) pair though spec only enforces c1–c4; risk of unintended over-constraint.
- **Stem unfreeze delayed by extra interval**; `training.py:778-784`; may slow full-network adaptation vs spec expectation.
- **No search result caching**; `gomoku.py:1061-1330`; increases compute cost relative to spec; impacts throughput.
- **Metrics bias by batch count**; `training.py:1148-1194`; losses averaged by number of batches instead of sample count, slightly biasing reported metrics (monitoring risk).
- **Win-rate not in search_training.csv**; `main.py:555-569`, `csv_logger.py:102-113`; metric exists elsewhere but not in primary search log (monitoring friction).
- **CUDA-only device selection**; `main.py:66`; risk of hard failure on non-GPU hosts.
- **Candidate generation assumes ample legal moves**; `gomoku.py:971-982`, `gomoku.py:1013-1019`, `training.py:1069-1137`; code assumes ≥6 legal moves so candidate lists are full and legal—likely true for typical 20–30 move games but could break in late-game/edge positions.

## 3) **Recommendation / nice to have**
- Add explicit guards for low legal-move counts (skip/pad candidates) and/or enforce legal-only candidate lists before search expansion if you want robustness beyond typical game lengths.
- Force optimizer recreation on resume or validate param coverage against current `requires_grad` set.
- Add a small deterministic negamax sanity test (single position) to lock perspective/sign handling.
- Clarify spec intent (k=3 vs k=6), and document any deliberate deviations (ranking scope, stem unfreeze timing).
- Log win rate directly in `search_training.csv` for a single source of truth.
