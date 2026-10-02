# Design Rationale

We trained a Gomoku policy AI model (15×15 board, no-restriction rules). The same network architecture went through two training routes: first search-free self-play reinforcement learning, then AlphaZero-style MCTS self-play training. The second route **does not inherit** the first route's parameters — it uses the first route's product as a teacher to distill a brand-new network that already knows how to play, and starts from that.

**This file is about "why it was designed this way."** General concepts and common practices are standard knowledge and are not defined anywhere in these documents; their Key Takeaways in `concept_list.md` say only why each one matters in this project. For this project's own designs (rows marked `our design` there), this file also says what the design is, but only as far as the "why" needs it. For implementation details (formulas, constant values, code organization), see each subdirectory's `CLAUDE.md`; below that, read the code directly — it's right here in the repo. Prerequisite relationships between concepts are also in `concept_list.md`.

Only concrete numbers that **themselves constitute an argument** are kept in the text (e.g., "a random game averages 109 moves" is the reason Tactical Boost exists); everything else is described only qualitatively, and the code is authoritative for actual values.

---

## 1. Training

### 1.1 Reinforcement Learning Pipeline

> Implementation details: `main/CLAUDE.md`. In `vanilla/` and `mini/`, every `.py` except `model.py` is a symlink to `main/`; the training recipe is identical.

#### 1.1.1 Self-Play Reinforcement Learning

The other path to training a board-game AI is supervised learning — learning from human game records. But that path has two fundamental limitations: (a) it needs a massive number of high-quality games; human game records are hard to obtain at scale, and they need labeling (which moves are good? which aren't?); (b) the ceiling of imitating humans is humans themselves — the AI can never surpass its training data. Self-play reinforcement learning sidesteps both problems: the model plays against itself (or its own past versions) to generate unlimited training data, without any human game records; the game outcome serves as the reward signal, letting the model discover superhuman strategies on its own through trial and error.

##### 1.1.1.1 Sparse Rewards

Gomoku's reward is extremely sparse: a game may last dozens or even over a hundred moves, but only the end of the game produces a reward signal (win +1, loss −1, draw 0); none of the intermediate steps get any immediate feedback. This means a good move and a bad move are completely indistinguishable at the reward level, and the gradient signal has a very low signal-to-noise ratio. Every design in this section is, at heart, an attempt to extract a finer-grained, per-step learning signal from this sparse terminal signal.

###### Actor-Critic and Value Baseline

Common practice. Sparse rewards require us to be able to evaluate each move step by step, and Actor-Critic measures action quality indirectly through the **difference** between position evaluations.

The reason for using position differences, rather than directly training a network to evaluate actions, is that the quality of a position has a natural supervision signal — the win rate of the games that continue from it — whereas the quality of a single action has no directly obtainable label. Translating "is this move good?" into "did the position get better or worse after playing it?" moves the problem onto the side that has labels.

---

###### GAE (Generalized Advantage Estimation)

Common practice. The core problem it solves is an inaccurate value network: the single-step TD advantage relies entirely on the value network's evaluation of the next position; if that evaluation is wrong, the signal is wrong.

GAE mitigates this by taking an exponentially weighted average of multi-step TD residuals. Its core assumption is that **a move's impact becomes gradually apparent in subsequent positions** — the further along, the clearer the situation and the more reliable the value network's evaluation. λ controls this time span: larger λ relies more on long-range signals (lower bias but higher variance); smaller λ relies more on near-term signals (lower variance but may miss effects that take hold later).

---

###### Baseline Ramp

Early in training the value head has not converged; its predictions are essentially noise. Relying on it for GAE at this point amounts to polluting the gradient signal with a wrong baseline.

So early on we use the game's actual outcome as ground truth instead, and switch fully to GAE once the value head is basically reliable, with a cosine curve smoothing the transition in between. Below, the position along this transition is written α: α=0 is pure raw return, α=1 is pure GAE.

---

###### Value Network Training (λ-return and Value Loss Coefficient)

> Note: the motivation for the Value Loss Coefficient involves the gradient ratio between the policy loss and the value loss; fully understanding it requires first understanding the multi-head design on the model-architecture side (Section 2.3).

Training the value network faces the same MC vs TD trade-off, but here the core dimension is not bias vs variance but **variance vs signal propagation speed**. TD(0) propagates extremely slowly: the terminal reward moves back only one step per update, and its strength halves at each step. A 30-move game needs about 30 updates to carry the signal back to the opening, and by then it has decayed to the order of 0.5³⁰ — nearly unusable. So multi-step returns are necessary to speed up propagation; experiments found that the λ currently in use is already enough to solve the propagation-speed problem, with no need to fall back to pure MC and its larger variance.

The Value Loss Coefficient decreases as the baseline α rises, to compensate for the change in policy-loss magnitude: at α=0 the advantage is the raw return (±1) and the policy loss is large; at α=1 the advantage becomes GAE (typical values much smaller than 1) and the policy loss drops accordingly. Only by adjusting the coefficient in step can the ratio between the policy and value gradients stay balanced.

---

###### Tactical Boost

This exists to speed up convergence **early** in training, not to strengthen play later — win-in-1 and block-in-1 are simple tactics the model can also learn on its own, so this mechanism has to be able to retire itself. The boost strength is therefore designed to adapt to the miss rate: the more misses, the heavier the boost; once the miss rate comes down, it automatically goes to zero.

The early-training predicament is twofold: the value network hasn't converged (its output is roughly noise), so only raw win/loss returns can be used; and the policy network is close to random — on a 15×15 board, a random game averages about 109 moves (about 20–30 moves in mid-to-late training), so the sparse reward is diluted to the extreme. Once synthetic samples directly teach win-in-1, games get markedly shorter and the signal density of the raw return rises accordingly, forming a positive feedback loop. The tactical search covers only the squares adjacent to existing stones, because in Gomoku a win-in-1 or block-in-1 move is necessarily adjacent to existing stones.

An edge case that is easy to get wrong: when the opponent has **two or more** independent win-in-1 threats at the same time, no "block-in-1" sample is produced. "Must block" means *play here or lose*; facing a double threat you can answer only one and lose either way, so no move can save the position, and there is nothing to teach. Counting such positions as "misses" would make the model train toward an unreachable target, and would also inflate the miss-rate statistic that drives the boost strength.

---

##### 1.1.1.2 Non-Transitivity of Win Rate

In two-player games, strength relations between strategies are often not transitive: A beats B, B beats C, but C may in turn beat A (like rock-paper-scissors). This means pure self-play (always playing against the latest version of itself) can make the strategy oscillate in cycles rather than improve monotonically — the model learns to counter the previous version but loses the ability to handle earlier ones. The designs in this section share one goal: making the model face diverse opponents.

###### Opponent Pool

The pool keeps multiple historical checkpoints from different periods as opponents, forcing the current model to stay robust against many styles rather than overfitting to the latest version.

Sampling half uniformly and half weighted by difficulty tilts the limited game budget toward the model's weaknesses — opponents it can't beat deserve a few more games. The win-rate threshold for entering the pool prevents a clearly weaker checkpoint from slipping in and dragging down the level of the whole pool.

---

###### Exploiter Mining

The pool's regular updates only add the current checkpoint at evaluation time, covering what is "recent." But some old checkpoints produced mid-training may happen to counter the current strategy — e.g., the current model has "forgotten" how to handle some early style of play. Mining goes back through history to find these exploiters, filling the pool's coverage blind spots.

A full rescan of history is unaffordable, so screening happens in two phases (a coarse screen with a few games, then a fine screen on the hardest few), and each pass scans only one shard of history, relying on rotation to guarantee long-term coverage.

---

###### KL-Aware Eviction and Mining

If eviction looked only at win rate, the opponents in the pool would become increasingly homogeneous — and since diversity is the pool's entire reason for existing, this would slowly hollow it out.

KL divergence measures the difference between two policy distributions; the smaller a member's KL distance to its nearest neighbor, the more redundant it is. Combining win rate with this redundancy measure when making decisions strikes a balance between "keeping hard opponents" and "keeping diversity": an opponent with a distinctive style is worth keeping even if it is easy to beat; one that is both easy and similar to others is evicted first. Mining works the same way — a candidate too similar to existing members must be harder to earn a place, while a candidate with a distinctive style faces a lower bar.

Comparing the "style" of two checkpoints requires a common baseline, so the current model first plays itself to generate a fixed set of probe positions, and then every member outputs its policy distributions on this set of positions — those distributions are the checkpoint's fingerprint.

---

#### 1.1.2 Exploration Enhancement

An RL agent faces the "exploration vs exploitation" dilemma: should it act according to the policy it currently believes is best (exploitation), or try uncertain new moves to discover potentially better choices (exploration)? Too much exploitation makes the policy settle prematurely into a local optimum; too much exploration wastes training resources on low-quality random moves.

##### 1.1.2.1 On-Policy Exploration

On-policy enhancement acts on the policy itself: it modifies the loss function directly so that the model is encouraged during training to keep a certain amount of randomness in its output. What it affects is the shape of the policy distribution — not too sharply concentrated on a single action.

###### Entropy Bonus and Reference Entropy

A fixed-strength entropy bonus isn't flexible enough: early in training, entropy is naturally high and needs no extra encouragement; later, the policy should become more deterministic, yet must not collapse completely.

Here we instead use a reference curve that decays with training progress, with the bonus strength proportional to the **ratio** of the reference to the current entropy. This division form provides strong anti-collapse protection — the lower the entropy, the larger the bonus's gradient with respect to entropy (the ratio tends to infinity as entropy→0), enough to stop the policy from sliding toward determinism.

But it is not a precise controller: there is no integral term, and no guarantee that entropy converges exactly onto the curve. That's why the curve is called a "reference" rather than a "target" — it constrains the order of magnitude, not the exact value.

---

##### 1.1.2.2 Off-Policy Exploration

Off-policy enhancement doesn't change the policy distribution itself; instead it injects signals from other sources into the training data — possibly better moves discovered through rollouts, or the opponent's good moves. These samples let the model learn moves its own exploration process is unlikely to reach.

###### Imitation Learning

This was introduced to fix a training collapse we actually ran into. In normal training, the model's win rate against the pool fluctuates between 50% and 60% (about 50% right after the latest checkpoint joins the pool; after a stretch of training it climbs to about 60%, and then the current checkpoint joins the pool); but in one run, the win rate plunged below 5% within a few dozen updates and never recovered.

Imitation provides a recovery channel: when the model gets weaker, it learns directly from the winning moves of stronger historical versions in the pool, getting the gradient signal it needs to climb out of the trough. Since it is a recovery mechanism, its weight should retire as the win rate recovers — learn more when weak, go to zero once strong.

Only moves from games the opponent **won as White** are imitated. Black moves first, and this first-move advantage is so large that "Black won" carries almost no information by itself — a Black win doesn't show that Black played well; a White win is the case truly worth learning from.

---

###### OPR (Off-Policy Rollout)

All the other exploration enhancements are essentially model-based — even Imitation learns the policy of historical checkpoints, which is on-past-policy rather than truly off-policy. OPR is the only model-independent exploration mechanism in this system: it evaluates alternative moves through random rollouts, and can therefore discover choices the model itself is unlikely to explore.

The cost is very low sample efficiency (random play rarely hits good moves), so it is used only on the errors most worth correcting: decision points in lost games where the model was overconfident (low entropy). Low entropy has a double meaning here: it indicates both that the model made a mistake at this point and that the model would have a hard time discovering that mistake through its own exploration. It also requires being far from the end of the game, to avoid the immediate tactical positions already covered by Tactical Boost.

---

#### 1.1.3 Generalization Enhancement

The distribution of the training data determines the boundary of the model's capabilities. If the openings and position structures of training games are too uniform, the model will perform well on familiar patterns but quickly fall apart in unfamiliar positions.

###### Renju Opening Seed

Initializing a portion of games with predefined Renju opening sequences is meant to keep the model from converging to fixed opening routines. There are two trade-offs here: the openings are used in only a portion of games because using them in all games would leave the model unable to learn to open on its own; and the Renju opening set is itself biased — not every Renju opening is sound, and not every sound opening is in the Renju set — but it is an easy-to-implement diversity source with broad enough coverage.

---

###### 8-Fold Dihedral Augmentation

Common practice. Gomoku's rules are completely invariant under the board's 8 dihedral transformations, so every sample can be expanded for free into 8 strictly equivalent samples: this both gives 8× data efficiency and forces the model to respect the board's symmetries: the policy must transform along with the board (equivariant), while the value must stay unchanged (invariant).

---

### 1.2 MCTS Self-Play Training

> Implementation details: `mcts/CLAUDE.md`

The RL-trained model is an "intuitive" player — it makes each move after a single forward pass, with no lookahead at all. To keep making it stronger, the natural next step is to introduce search: let the model "think" at each move (using tree search to simulate how various moves play out), then learn the better decisions found by search back into the network, so that even without search it can make decisions close to search level.

This route completely abandons the entire RL machinery — no entropy bonus, no GAE, no Tactical Boost, no Imitation, no OPR, not even an opponent pool; it plays only itself. The reason is simple: the search itself supplies exploration, and the search results themselves are the labels.

#### 1.2.1 Warm Start

###### From Post-Training to Distilled Warm Start

The first-generation approach was true "post-training": inherit the RL model's parameters, switch the training target to MCTS search results, and keep training. It hit the wall of distribution-shape mismatch — the MCTS visit distribution is naturally flatter than a policy distribution (search spreads visits across multiple branches), while the RL-trained policy is quite sharp. Inheriting parameters meant the network's output distribution was off from the supervision target by a whole shape right from the start; the initial loss was huge and the model oscillated. This wasn't because it was learning poorly — the two distributions simply had different shapes.

The first generation used a temperature parameter T to **compensate** for this difference: before search, soften the model prior to the MCTS entropy level; after search, sharpen the MCTS output back to the model's entropy level, so that the target the model sees is aligned in entropy with its own output and it only has to learn the relative changes MCTS discovers. This compensator had two problems. First, it was **not self-stabilizing**: pure compensation has no restoring force, so any drift in model entropy is absorbed by T rather than corrected, and T could drift all the way to divergence (the countermeasure at the time was to make the sharpening slightly incomplete, creating a bit of restoring force to pull T back toward 1). Second, **the optimal scaling ratio varies by position** (it differs between the opening, the middlegame, and the endgame), yet it cannot be solved per position: PUCT needs T as an input, while computing T for a position requires that position's PUCT result — a circular dependency — so it could only fall back to an imprecise global average.

The current generation no longer compensates for this difference; instead it **eliminates it at the source**: a freshly initialized network sees only MCTS-shaped targets from its very first step. The shape-mismatch problem no longer exists, so the compensator is naturally no longer needed.

So "not inheriting parameters" is not a compromise or a side effect — **it is the solution itself**; inheriting parameters is exactly why the first generation failed. For the same reason, this route should not be called "post-training" or "fine-tuning": there is no parameter inheritance whatsoever between the two routes' checkpoints. The interface between the two training routes is a **dataset**, not a weights file; what is passed on is **behavior**, not parameters.

---

###### The Three-Stage Pipeline

The whole route has three stages, numbered as in the code: in stage 0 (`generate_data`), a frozen RL teacher runs MCTS self-play, writing each move's position and search result to disk; in stage 1, a freshly initialized student is distilled offline on this static data, without running any search; in stage 2, this student starts searching on its own, playing itself, and training itself on its own search results. Only stage 2 is real MCTS training; stages 0 and 1 exist solely to build its starting point.

Why is such a starting point needed? A randomly initialized network gives nearly uniform priors and meaningless leaf evaluations; MCTS at that point would waste its simulation budget on uninformative exploration, and would then train the network on nearly random visit distributions. The warm start skips this whole phase: the RL checkpoint already knows how to play, so the very first search in stage 2 is already informative.

Why is the warm start itself split into two stages? Because the RL checkpoint is a **policy network**, not an MCTS-shaped target. Stage 0 translates it into a search-shaped supervision signal, so that stage 1 can align the student to that shape.

There is an easily overlooked but crucial point here: what the student learns is **not the teacher's policy but the "teacher + search" policy** — it starts out one step ahead of the teacher. But what it is ahead of is the **teacher**; it is not the fixed point of **its own** search. The next section relies on this.

---

#### 1.2.2 The Training Loop and When to Stop

###### MCTS as a Policy-Improvement Operator

Think of MCTS as an **improvement operator** acting on the network's raw policy: given the network's raw policy, search outputs a better policy; each round of training distills the raw policy onto its own search results, iterating repeatedly toward the fixed point that "even search can't improve." The gap between the two — the KL between the search result and the raw policy — is an observable measure of this improvement.

This framework explains in one stroke two things that otherwise look unrelated.

First, **the loss spike at the start of stage 2**. Stage 1 only got the student aligned with "teacher + search"; it is ahead of the teacher, but has never been at the fixed point of its own search. So at the start of stage 2, the target becomes "the student itself + search," which opens up a brand-new gap that has never been distilled — that is the spike; subsequent iterations push it back down to a slow steady-state drift. This also shows that the target early in training is strongly non-stationary: a target saved one round earlier is already stale a few updates later.

Second, **it provides a progress signal that doesn't depend on the optimizer**. This KL measures the gap between search and network, not "how well the network fits the current target." The latter mixes optimization progress with the improvement signal, and so can't be used to judge whether training should continue or stop.

---

###### Plateau-Driven Staircase LR

Stage 2 has **no preset cap on training steps**. The learning rate stays constant until the improvement measure above stops declining and plateaus, at which point the learning rate is multiplied by a factor less than 1; after a certain number of drops, the next plateau ends training.

Per the framework in the previous section, the stronger the network, the closer its raw policy is to the fixed point, and the smaller the improvement. But it can't drop to 0; two "floors" hold it up: one is **finite policy capacity** (the network may not be able to represent the search output exactly), the other is **optimization noise injected by too large a learning rate**.

When a plateau appears, there is no way to tell directly which kind the remaining gap belongs to. Lowering the learning rate is precisely a test of the latter hypothesis: with less noise, if the curve starts declining again, it was indeed being held up by noise, and that is a gain; if the curve still doesn't move, it was already close to the capacity floor, and little is lost. This is a test with asymmetric costs, so it is worth running — and for the same reason, "when to stop" doesn't need a manually chosen step count; this signal provides it on its own.

---

#### 1.2.3 Getting More Out of Each Search

###### Subtree Harvesting

Each move's search evaluates thousands of leaf positions, but by default only the root position that is actually played becomes a training sample; everything else is discarded. Besides the root node's conclusion, the output of a single search also contains a large number of conclusions about child positions whose statistics are already fairly sufficient — not harvesting them means that computation was wasted.

The weighting and the thresholds both come from the same consideration: how trustworthy a node's statistics are depends on how many times it was visited. So sample weight rises with visit count, root nodes searched with the full budget automatically dominate, and shallow nodes serve only as a small supplement. Moreover, value targets and policy targets have different trust thresholds — a value target is just a mean (a low-order statistic) and becomes reliable earlier than an entire policy distribution (a high-order statistic) — so the two use different visit-count thresholds, and nodes that clear only the value threshold participate only in the value loss.

---

###### Action Temperature and Harvesting

When choosing moves during self-play, we don't sample directly from the visit distribution; instead we first sharpen the distribution with a temperature below 1. The lower the temperature, the stronger the moves played, so games stay on higher-quality lines and reach deeper, more complex positions — which is where training signal is richest. The cost is trajectory diversity: a sharp policy replays the same few lines over and over, narrowing the data's coverage.

Subtree harvesting buys exactly this diversity back: each move contributes not only the root node that was actually played, but also a handful of internal nodes that **deviate from the game's main line**. So it is precisely harvesting that makes a temperature below 1 viable here — the two designs go together, and looked at in isolation, either one would seem to have an obvious flaw.

---

#### 1.2.4 Inside the Search

###### Backup Discount

When search results are backed up along the path, at each level up the value is not only negated for the change of perspective but also multiplied by a discount slightly less than 1.

Without a discount, all losing lines look completely equivalent to the search — losing in five moves and losing in twenty are the same −1. With a discount, the later an outcome occurs, the smaller its influence, so "losing later" is strictly better than "losing earlier," and "winning earlier" is strictly better than "winning later." This gives the same terminal outcome an internal preference ordering: when the position is unfavorable, search tends to push the loss further away (which is usually exactly the stubborn, trouble-making style of defense), and when it is favorable, search tends to close out quickly rather than pointlessly dragging things out.

---

## 2. Architecture

### 2.1 Shared Design

> Implementation details: `main/CLAUDE.md`; for the input/output conventions, see the root `CLAUDE.md`

###### Canonical Board Representation

Gomoku's first-move advantage is huge; in training, Black's win rate is often above 70%. If the model could tell which color it is playing (e.g., via an extra input channel), it might learn to play passively as White — it's probably going to lose anyway.

The canonical representation gives the model no explicit color signal: the input is always encoded from the perspective of the "side to move"; it sees only "own stones" and "opponent stones," and the value output is always the expected return from the side to move's perspective. (Color is still inferable in principle — with Black to move both sides have the same number of stones, with White to move the mover has one fewer — but the model would have to learn to count to get it.) Playing Black and playing White are pushed toward sharing the same policy.

---

### 2.2 Vanilla and mini Models (Control Group)

> Implementation details: `vanilla/CLAUDE.md`

These two models exist as **controls**, not as alternatives. The three training directories share the same training recipe (every file except `model.py` is a symlink to the same file), so differences in playing strength among the three can be cleanly attributed to the model itself: `vanilla` vs `main` isolates the "architecture design" variable, and `vanilla` vs `mini` isolates the "parameter scale" variable. If the training recipes differed too, there would be no way to tell where a difference came from.

---

### 2.3 Main Model

> Implementation details: `main/CLAUDE.md`

###### Multi-Scale Directional Stem

Gomoku's core threats (open threes, fours, fives) lie strictly along 4 directions. A standard N×N convolution does cover these directions, but it also includes many "off-axis filler" positions — of the 25 weights in a 5×5 kernel, many positions are useless for directional patterns — which both wastes parameters and requires more data to learn to ignore them. A directional kernel with nonzero weights only on the 4 lines is a strong inductive bias that matches Gomoku's geometry exactly.

Approximating directional kernels with "a sum of several 3×3 convolutions with different dilation rates," rather than using true 1×N kernels, is a purely engineering consideration: PyTorch/cuDNN are highly optimized for 3×3 convolutions, and non-standard kernel shapes actually run slower. The center position of the high-dilation branches is forced to zero, to avoid covering the same point twice with the dilation-1 branch.

---

###### Trunk Dilation Schedule

The board is 15×15, and long-range spatial relationships (e.g., interplay between distant threats) need a large receptive field. With plain dilation-1 3×3 convolutions, the receptive field grows by only 2 per layer; even stacking the trunk's full depth, the theoretical receptive field barely covers the board, and the effective range is much smaller in practice.

Interleaving larger dilation rates quickly expands the effective receptive field, while a dilated convolution has exactly the same parameter count and compute as a plain 3×3 — an almost free change.

---

###### Late-Branching Dual-SE Design

The policy head and the value head care about different things — the policy about local tactics and specific move locations, the value about the overall balance of the position — and that is why the two heads are built differently, each with an inductive bias suited to its task (see Dual-Attention Policy Head and Value Head below). The trunk's job, however, is common to both: extracting spatial features (stone arrangements, basic patterns). The traditional approach is a hard split at some depth of the trunk, which lands in a dilemma: branch early, and each task has to learn its own copy of the spatial-feature weights from that point on; branch late, and the two tasks interfere with each other in the shared layers.

The dual-SE blocks at the end of the trunk sidestep this dilemma with a soft branch. The two streams keep sharing the convolution weights — spatial feature extraction is the common task, and having both tasks' gradients train the same weights also acts as an implicit regularizer: fewer free weights means less self-play data is needed to train them — while divergence comes only from each stream's own SE gating. SE has far fewer parameters than a convolutional layer, so the divergence costs almost no extra weights (compute is not saved: each stream still runs its own pass through the shared convolutions), and it is **gradual**: a little more divergence with each block, rather than a single clean cut. The normalization layers are also made separate, because differences in normalization statistics are another dimension of divergence that costs almost no parameters.

---

###### Dual-Attention Policy Head

Gomoku move decisions often require global reasoning — e.g., creating a threat at a distant point to support the current attack, or comparing the priorities of several candidate points across the board. Pure convolutions have a limited receptive field, which may not suffice even with the trunk's dilated convolutions. Self-attention provides global information exchange at O(N²) cost; the board has only 225 positions, so N² = 50625, an acceptable amount of compute. Sandwiching a convolutional refinement between two attention layers lets the model first do one round of global reasoning, then tidy things up locally with convolution, and then do a second round of global reasoning.

Two details also come from design considerations. First, attention needs to know the relative relationship between positions, and the board's dihedral symmetry means this relationship depends only on the larger and the smaller of the absolute row and column offsets — exploiting this, covering all position pairs takes only 120 learnable parameters, rather than independent parameters for every pair. Second, the output projection is zero-initialized, so attention is approximately an identity map early in training: attention starts out "doing nothing," and training decides how much it should do, rather than injecting random global mixing into the features from the start.

---

###### Value Head

The core idea is "expand channels → spatial pooling → fully connected." Expanding channels lets each channel represent a board pattern (e.g., a particular open three or a particular four); spatial pooling provides **position invariance** — wherever on the board a threat appears, it means the same thing for position evaluation, and pooling turns "where a pattern is" into "whether the pattern is present."

Using log-mean-exp pooling instead of ordinary global average pooling is a signal-to-noise consideration: the board has 225 points, and when detecting sparse features, average pooling drowns the signal from a few high-response positions in the noise from a large number of low-response positions. Log-mean-exp is a learnable interpolation between average pooling and max pooling (as the temperature goes to infinity it degenerates to the average, and as it goes to zero, to the max); each channel learns its own temperature, and channels that detect sparse patterns automatically lean toward max pooling to preserve the signal.
