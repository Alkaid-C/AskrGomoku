# Concept Dependency List

## Gomoku Basics

| # | Type | Concept | Prerequisites | Key Takeaway |
|---|---|---|---|---|
| G1 | concept | Basic Rules | — | 15×15 board, black goes first, 5 in a row along any of 4 directions (horizontal, vertical, diagonal, anti-diagonal) wins; this project uses no-restriction rules |
| G2 | concept | Zero-Sum Game | — | Definition; knowing one side's evaluation determines both |
| G3 | concept | Basic Tactics | G1 | Open three, open four, half-open four; threats are strictly directional; a single threat can always be blocked; fork (two simultaneous must-block threats) is the only way to force a win |
| G4 | concept | First-Player Advantage | G1 | Human experts playing black achieve 100% win rate; structural property of the game, not a skill gap |
| G5 | concept | Renju Opening | G1 | Definition |

## Neural Network

| # | Type | Concept | Prerequisites | Key Takeaway |
|---|---|---|---|---|
| N1 | concept | Neural Network | — | A mathematical function with many tunable parameters; training: given input and desired output, adjust parameters to fit; inference: given input and parameters, compute output |
| N2 | our design | Basic I/O Format (C0, C1, policy) | N1, G4 | Input: C0 = my stones, C1 = opponent's stones, flipped every move; output: probability distribution over legal positions; value and C2 deferred to N10/N14; in design rationale (2.1 Canonical Board Representation) |
| N3 | concept | FC | N1 | Inspired by biological neurons (y = wᵀx + b); defines a hyperplane — a linear classifier (Perceptron) |
| N4 | concept | Activation | N3 | Activation introduces nonlinearity; projects data into a space where linearly inseparable problems become separable (Cover's theorem intuition) |
| N5 | concept | MLP | N3, N4 | FC + Activation stacked alternately; theoretically can approximate any continuous function (universal approximation) |
| N6 | concept | Feature Space | — | Behind raw data lie latent features (e.g., "is there an open three here?"); we operate on features, not raw data; the space of meaningful states is far smaller than the raw dimensionality suggests (intrinsic dimensionality) |
| N7 | concept | Inductive Bias | N1 | Adds equalities/constraints (via architecture, data augmentation, etc.) that reduce degrees of freedom; if the constraints match reality, the model learns faster with less data |
| N8 | concept | CNN | N5, N6, N7 | Prior: the same local pattern matters regardless of where it appears on the board; FC with two constraints: (1) distant connections dropped, (2) same weights at every position; input is a tensor of height × width × channels; drastically fewer parameters than MLP |
| N9 | concept | Padding for CNN | N8 | Adds border around input so the filter can process edge positions; without padding, output shrinks and edges are underrepresented |
| N10 | our design | Third Input Channel (C2) | N2, N9 | In the root `CLAUDE.md` (model input convention) |
| N11 | common practice | Skip Connection | N8 | Each layer adds a modification to its input rather than replacing it: h_{i+1} = h_i + C(h_i); gradient becomes a sum over all orders instead of a single high-order product — low-order terms prevent vanishing gradient; Stem = layers before residual blocks, Trunk = stack of residual blocks, Head = layers after trunk |
| N12 | common practice | Norm | N11 | Rescales intermediate activations to stable range; prevents signals from exploding or vanishing during training |
| N13 | common practice | Policy Head of Vanilla | N2, N11 | Convolutional layers compress trunk features down to 1 channel; each spatial position directly corresponds to a board position's logit; softmax converts to probability distribution |
| N14 | common practice | Value Head of Vanilla | N2, N5, N11 | 1×1 conv reduces channels, then flatten + FC maps spatial features to a single scalar; tanh bounds output to [-1, 1] representing expected return from current player's perspective |
| N15 | our design | Multi-Scale Directional Stem | N8, N7 | In design rationale (2.3 Multi-Scale Directional Stem) |
| N16 | our design | Trunk Dilation Schedule | N11, N8 | In design rationale (2.3 Trunk Dilation Schedule) |
| N17 | common practice | Hard Parameter Sharing | N11, T9 | Definition; shared layers receive gradients from both tasks — more signal per sample; assumes low-level features (board patterns) are useful for both tasks |
| N18 | concept | Gradient Conflict | N17, T3 | When two loss terms share parameters, their gradients may point in opposing directions; one task's improvement can degrade the other; how many parameters should be shared is an open question |
| N19 | common practice | Value Loss Coefficient | N18 | A scalar multiplier on value loss to balance gradient magnitudes between policy and value; prevents one task from dominating shared layer updates |
| N20 | common practice | Late-Branching | N17, N18 | Later layers handle more task-specific features; branching late lets both tasks share low-level feature extraction while separating high-level reasoning |
| N21 | common practice | SE | N8, N6 | Squeeze-and-Excitation: global average pool → FC → sigmoid produces per-channel gate; lets the network learn "which channels matter for this input"; lightweight (few parameters relative to conv layers) |
| N22 | our design | SE&Norm Branching | N20, N21, N12 | In design rationale (2.3 Late-Branching Dual-SE Design) |
| N23 | concept | Transformer | N5 | Definition; enables global reasoning across all positions |
| N24 | our design | Policy Head of Main | N13, N23, N7 | In design rationale (2.3 Dual-Attention Policy Head) |
| N25 | our design | Value Head of Main | N14, N6 | In design rationale (2.3 Value Head) |
| N26 | our design | Model Comparison (vanilla / main / mini) | N8 | In design rationale (2.2 Vanilla and mini Models) |

## Training

| # | Type | Concept | Prerequisites | Key Takeaway |
|---|---|---|---|---|
| T1 | concept | Machine Learning | — | Definition |
| T2 | concept | Loss | T1 | Definition |
| T3 | concept | Gradient Descent | T1, T2 | Definition |
| T4 | concept | Supervised Learning | T2, T3 | Definition; requires labeled data |
| T5 | concept | Reinforcement Learning | T2, T3 | Definition; no labeled correct action — only outcome feedback |
| T6 | common practice | Self-Play | T5 | Definition; not bounded by human skill ceiling |
| T7 | concept | Monte Carlo Return | T5 | Definition; unbiased but high variance |
| T8 | concept | Sparse Rewards | T5 | A game may last 20-100+ moves but reward only arrives at the end; intermediate moves get no direct feedback; a good move and a bad move look identical in terms of reward signal |
| T9 | common practice | Actor-Critic | T7, T8 | Two models: actor (policy — picks moves) and critic (value — evaluates positions); turns sparse game-end reward into dense per-move signal; cost: an additional model to train; risk: critic inaccuracy introduces bias |
| T10 | concept | Value Model | T9 | Definition; the "critic" in actor-critic; evaluates positions (states), not actions directly — action quality is inferred from the change in position evaluation |
| T11 | common practice | Value Baseline (TD(0)) | T10 | Advantage = V(next state) - V(current state); relies entirely on the critic's accuracy; low variance but biased when critic is inaccurate; signal propagates only one step per update |
| T12 | common practice | GAE | T7, T11 | Blends MC return (long-horizon, high-variance) and TD(0) (short-horizon, low-variance) via parameter λ; λ=1 is pure MC, λ=0 is pure TD(0); lets reliable long-term outcomes correct short-term critic errors |
| T13 | our design | Ramp of GAE | T7, T12 | In design rationale (1.1.1.1 Baseline Ramp + 1.1.1.1 Value Network Training) |
| T14 | our design | Tactic Boost | T8 | In design rationale (1.1.1.1 Tactical Boost) |
| T15 | concept | Non-Transitivity of Win Rate | T6 | A beats B, B beats C, but C may beat A (rock-paper-scissors); strength is not a single ranking; pure self-play can cycle without improving |
| T16 | common practice | Opponent Pool | T6, T15 | Maintain a pool of past checkpoints as opponents; forces the model to stay robust against diverse play styles rather than over-adapting to a single opponent; our specific sampling/admission choices in design rationale (1.1.1.2 Opponent Pool) |
| T17 | our design | Exploiter Mining | T16 | In design rationale (1.1.1.2 Exploiter Mining) |
| T18 | concept | KL-Divergence | T21 | Definition; used here to quantify style difference between opponents |
| T19 | our design | Divergence-Aware Eviction and Mining | T16, T17, T18 | In design rationale (1.1.1.2 KL-Aware Eviction and Mining) |
| T20 | concept | Exploration | T5 | Explore (try uncertain moves to discover better strategies) vs exploit (play the current best move); too much exploitation → stuck in local optimum; too much exploration → wasted training |
| T21 | concept | Entropy | — | Definition; maximum entropy for 225 positions ≈ 5.42 nats |
| T22 | common practice | Entropy Bonus | T20, T21 | Add a reward for keeping entropy above zero in the loss function; prevents the policy from collapsing to a single deterministic move; maintains exploration capacity |
| T23 | our design | Dynamic Entropy Bonus (Reference Entropy) | T22 | In design rationale (1.1.2.1 Entropy Bonus and Reference Entropy) |
| T24 | our design | Opponent Imitation | T6, T8, G4 | In design rationale (1.1.2.2 Imitation Learning) |
| T25 | our design | Off-Policy Rollout | T20, T21, T8 | In design rationale (1.1.2.2 OPR) |
| T26 | concept | Generalization | T1 | Definition; both training data diversity and model design affect how well the model handles novel situations |
| T27 | common practice | Opening Seed | T6, T26 | Start some games from predefined opening positions instead of empty board; increases diversity of training positions; prevents the model from converging to a single opening repertoire |
| T28 | our design | Renju Opening Seed | T27, G5 | In design rationale (1.1.3 Renju Opening Seed) |
| T29 | common practice | 8-Fold Dihedral Augmentation | T26 | The board has 8 symmetries (4 rotations × 2 reflections); each training sample is expanded to 8 equivalent samples; 8× data efficiency and forces the model to learn rotationally invariant features |
| T30 | common practice | MCTS (PUCT version) | T4, T10 | Tree search: repeatedly simulate future moves, using the neural network to evaluate leaf positions and guide which branches to explore; after many simulations, the visit distribution is a better decision than raw policy; PUCT formula balances exploiting high-value branches and exploring under-visited ones |
| T31 | common practice | Knowledge Distillation | T4 | Train a student on a teacher model's outputs instead of ground-truth labels; the teacher's full output distribution carries far more information per sample than a single hard label |
| T32 | our design | Distilled Warm Start | T30, T31, T6 | In design rationale (1.2.1 Warm Start) |
| T33 | our design | MCTS as Policy-Improvement Operator | T30, T18 | In design rationale (1.2.2 MCTS as a Policy-Improvement Operator) |
| T34 | our design | Plateau-Driven Staircase LR | T33, T3 | In design rationale (1.2.2 Plateau-Driven Staircase LR) |
| T35 | our design | Subtree Harvesting | T30 | In design rationale (1.2.3 Subtree Harvesting) |
| T36 | our design | Action Temperature vs Trajectory Diversity | T30, T35, T20 | In design rationale (1.2.3 Action Temperature and Harvesting) |
| T37 | our design | Backup Discount | T30 | In design rationale (1.2.4 Backup Discount) |
