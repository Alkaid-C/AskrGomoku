# flip_analysis/ — Does MCTS Change the Network's Top-1?

Side study, not part of the pipeline. Stage 2's `avg_raw_mcts_kl` averages a few
tenths of a nat, but in many positions it is near zero and MCTS's top-1 equals the
raw top-1. Where does search actually override the network, and is the stage-2
simulation budget too small or too large?

```bash
# from mcts/ (paths resolve relative to it)
python3 flip_analysis/run_flip_analysis.py
python3 -m unittest flip_analysis.test_flip_tracker
```

- `run_flip_analysis.py` — self-plays a checkpoint from three start families
  (empty / Renju opening / random 3 stones), one batched search per ply, streaming
  one JSON row per ply to `flip_data.jsonl`. Knobs are argparse flags.
- `search_with_snapshots.py` — `mcts.mcts_search_batched`'s loop specialized for
  this study (no Dirichlet, raw priors, no harvesting), reusing `mcts.py`'s leaf
  eval and backup unchanged.

**Two budgets.** Each ply searches to `--total-sims` but plays a move sampled from
the visit distribution at `--action-sims`, so the trajectory is what the deployment
budget produces while flips are observed past it.

**Flip** = significant change of pairwise preference, not of instantaneous visit
argmax. Winner starts at raw argmax with the raw prior as baseline; a tied-for-top
candidate flips it when `(v(c)−v(w)) − (baseline(c)−baseline(w)) ≥ --margin`. Each
flip resets the baseline to the current visit distribution, so later flips need
fresh movement. Tracking starts at the first sim with `1/t < margin`, below which
visit quantization alone would manufacture flips.

Per ply it records the flip trajectory (`flip_sims`, `flip_events`, …), the moves
and their raw/search masses, and position features (`stone_count`, `steps_left`,
`color`, `outcome`, `raw_entropy`, `raw_value` vs `mcts_value`, `raw_mcts_kl_*`).
