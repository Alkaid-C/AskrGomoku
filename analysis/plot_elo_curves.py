"""Plot the three Elo curves from Q2 results."""

import os

import matplotlib.pyplot as plt
import numpy as np

D = np.load(os.path.join(os.path.dirname(__file__), 'elo_q2.npz'))
pip = D['pipelines']
upd = D['updates']
elo = D['elo']

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'mini': '#888888', 'vanilla': '#1f77b4', 'main': '#d62728'}
for name in ('mini', 'vanilla', 'main'):
    mask = pip == name
    order = np.argsort(upd[mask])
    x = upd[mask][order]
    y = elo[mask][order]
    ax.plot(x, y, label=f'{name} (final={y[-1]:+.0f})', color=colors[name], linewidth=1.5)

ax.set_xlabel('training update')
ax.set_ylabel('Elo (mean-zero across all 384 ckpts)')
ax.set_title('Elo curves — mini vs vanilla vs main\n'
             '32 openings × 384·383 ordered pairs, T=1.0, '
             'BT-MLE on win+0.5·draw')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

out = os.path.join(os.path.dirname(__file__), 'elo_curves.png')
plt.tight_layout()
plt.savefig(out, dpi=140)
print(f'Saved: {out}')
