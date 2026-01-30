### **Multi-Stage Review Plan for Negamax Post-Training**

**Overall Goal:** To verify that the implementation described in `IMPLEMENTATION_NOTES.md` correctly and efficiently realizes the system designed in `POST_TRAIN_SPEC.md`.

---

#### **Stage 1: Code Cleanup and Refactoring Verification**

*   **Objective**: Confirm that RL-specific enhancements have been cleanly removed or converted into non-intrusive probes.
*   **Files to Review**: `enhancement.py`, `training.py`, `main.py`, `csv_logger.py`.
*   **Key Review Points**:
    1.  **Code Removal**: In `enhancement.py`, verify that functions, classes, and constants related to Off-Policy Rollout (OPR) and training-time tactical boosts are fully removed as documented.
    2.  **Probe Implementation**: Review the new `probe_tactical_accuracy` function in `enhancement.py` to ensure it only calculates statistics and does not modify training data.
    3.  **Upstream Integration**: Check `training.py` and `main.py` to confirm that all calls to the removed functions are gone and that `probe_tactical_accuracy` is integrated correctly for metric gathering.
    4.  **Logging Schema**: In `csv_logger.py`, confirm that `training_updates.csv` columns are simplified and that the new `tactical_probe.csv` is defined with the correct schema.

---

#### **Stage 2: Negamax Search Implementation Review**

*   **Objective**: Scrutinize the core `negamax_batched` implementation for correctness, efficiency, and adherence to the specification.
*   **Files to Review**: `gomoku.py`.
*   **Key Review Points**:
    1.  **Candidate Generation**:
        *   Confirm `generate_candidates` produces 6 candidates for the root and 5 for internal nodes.
        *   Verify the logic for selecting the random neighbor candidate (Chebyshev distance ≤ 1).
    2.  **Search Logic**:
        *   Analyze `negamax_batched` to ensure it correctly implements the negamax algorithm (alternating perspectives, negating values) to the specified depth of `d=3`.
        *   Check that leaf nodes are always evaluated by the `current_model`'s value head.
        *   Confirm that terminal states (win/loss) are handled with early termination, returning +1 or -1.
    3.  **Batching Strategy**: Review the search implementation to ensure it expands the tree layer-by-layer and performs model inference in large batches, as described in the notes, to maximize GPU utilization.
    4.  **Move Sampling**: In `sample_move_from_q`, verify the implementation of the top-5 candidate sampling, including the Q-value scale normalization and the temperature-based softmax.

---

#### **Stage 3: Search-Based Self-Play Loop Verification**

*   **Objective**: Ensure the self-play process correctly uses the negamax search to generate games and `SearchSample` data.
*   **Files to Review**: `gomoku.py`.
*   **Key Review Points**:
    1.  **Data Structure**: Check that the `SearchSample` dataclass matches the specification, containing fields like `sorted_candidates`, `all_candidates`, and `V_target`.
    2.  **Data Generation Loop**: In `play_episodes_with_search`, trace the logic for a single move to confirm it: generates candidates, runs search, sorts results, records a `SearchSample` (only for the `current_model`), and samples the next action.
    3.  **Target Value**: Verify that the `V_target` field in each `SearchSample` is correctly assigned the maximum Q-value from the search results (`max(Q_search)`).

---

#### **Stage 4: Supervised Loss Function Review**

*   **Objective**: Verify the correctness of the custom ranking and value loss functions.
*   **Files to Review**: `training.py`.
*   **Key Review Points**:
    1.  **Ranking-Inside Loss**: Review `compute_ranking_inside_loss` to ensure the dynamic margin `min(m_rank, Q_norm_diff)` is implemented correctly for the top 4 candidates.
    2.  **Separation-Outside Loss**: Review `compute_separation_outside_loss` to confirm it correctly penalizes non-candidate moves for having logits higher than the 4th-ranked candidate (`c4`), and that it uses a `mean` aggregation.
    3.  **Value Loss**: Confirm `compute_search_value_loss` is a standard Mean Squared Error between the predicted value and the `V_target`.
    4.  **Total Loss**: Check that the final loss in `train_on_search_samples` is a weighted sum of the policy and value losses and that no entropy loss is included.
    5.  **Data Augmentation**: Verify that the 8-fold symmetry augmentation is correctly applied to both observations and candidate action indices via `augment_candidates_8fold`.

---

#### **Stage 5: Progressive Unfreezing Schedule Verification**

*   **Objective**: Ensure the parameter freezing and unfreezing logic correctly manages the training schedule.
*   **Files to Review**: `training.py`.
*   **Key Review Points**:
    1.  **Schedule Logic**: Review `apply_freeze_schedule` to confirm it correctly maps the current `update` number to the set of unfrozen parameters, following the sequence: heads → trunk blocks (15 down to 0) → stem.
    2.  **Optimizer Management**: Analyze `maybe_update_optimizer` to ensure it correctly detects a change in the unfreezing schedule and creates a new optimizer containing only the newly-trainable parameters. This is critical for the schedule to take effect.

---

#### **Stage 6: Final Integration and Logging Review**

*   **Objective**: Review the final integration into the main training loop and ensure all new metrics are being logged correctly.
*   **Files to Review**: `main.py`, `csv_logger.py`.
*   **Key Review Points**:
    1.  **Main Loop Integration**: Based on the usage example, review the `main.py` training loop to confirm that `play_episodes_with_search`, `train_on_search_samples`, and `maybe_update_optimizer` are called in the correct order.
    2.  **State Management**: Check that the `training_state.json` is being updated with necessary information like `unfrozen_blocks` to allow for correct resumption of training.
    3.  **Metrics Logging**: Confirm that the new `search_training.csv` is being populated with all the specified metrics (e.g., `top1_acc`, `ranking_inside_loss`, `unfrozen_blocks`) by the `log_search_training` method.
