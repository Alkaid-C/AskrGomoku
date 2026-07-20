# LR stair abolition

Counterfactual branch from the completed update-120 state. It rejects the LR
drop selected at update 120, keeps stair 2's LR fixed, and stops after update
152 so both branches can be compared over a full post-branch window.

The source checkpoint is copied without modification. `run_fixed_lr.py`
overrides only the loaded controller state; model weights, Adam moments, replay
buffer, and saved RNG states all come directly from the source snapshot.

Verify without starting training:

```bash
python3 lr_stair_abolition/run_fixed_lr.py --verify-only
```

Start or resume the experiment from the `mcts/` directory:

```bash
nohup python3 lr_stair_abolition/run_fixed_lr.py > lr_stair_abolition/RRR.log 2>&1 &
```

The base trainer's final message says "Final plateau reached" whenever its
controller sets `finished`; for this branch that line means the fixed horizon
was reached, not that a plateau test fired.
