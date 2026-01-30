# 五子棋 Top-k Negamax 后训练实现方案

## 1. 概述

对现有 RL checkpoint 进行后训练，使模型输出更适合 top-k negamax 搜索的使用方式：
- **Policy head**：提供可靠的 move ordering（候选排序）
- **Value head**：提供准确的 leaf evaluation（叶子估值）

后训练使用搜索产生的监督信号，不再使用 RL policy gradient。

## 2. 部署规格

| 参数 | 值 |
|------|-----|
| 搜索算法 | top-k negamax |
| 搜索深度 | d = 3 |
| 分支因子 | k = 3 |
| 叶子评估 | value head 输出 |

部署侧仅依赖 logits 的相对顺序（用于 top-k 选取），不依赖概率采样。

## 3. 数据生成

### 3.1 候选集生成

对每个决策状态 s，生成 6 个候选动作：

1. **Top-5**：policy logits 最高的 5 个合法动作
2. **随机 1 个**：从 Chebyshev distance ≤ 1 的邻域空位中随机选择（排除已选的 top-5）；若邻域无剩余空位则回退为随机合法动作（同样排除 top-5）

候选集 `C(s)` 规模固定为 6，且保证 6 个动作互不重复。

### 3.2 Top-k Negamax 搜索

在根节点状态 s 上运行 negamax（深度 d=3）：

- **分支生成**：每个节点通过 3.1 的方式生成 6 个候选（当前方回合使用 current model，对手回合使用 opponent model）
- **叶子评估**：深度耗尽时始终由 current model 的 value head 评估（从该节点行动方视角）
- **终局处理**：一旦出现胜负则早停，回传终局价值（+1 赢 / -1 输）
- **备份规则**：标准 negamax 备份（换手取负）
- **批量搜索实现**：采用非递归实现以提升 GPU 效率。先批量生成所有 k^d 个叶子状态，通过 model 一次（或少数几次）大批量推理获取 value，再反向传播回根节点。避免递归调用带来的频繁小批量推理开销。
- **缓存策略**：记录每个缓存状态由哪个 move 生成；当实际落子与缓存的 move 一致时，复用该子树的搜索结果。

搜索输出：
- `best_move(s)`：根节点最优动作，用于 self-play 实际落子
- `Q_search(s, a)`：根节点每个候选动作的搜索评分

### 3.3 候选排序

按 `Q_search` 降序排列候选动作：
```
c1, c2, c3, c4, c5, c6
```
其中 c1 的 Q_search 最高（对当前行动方最有利），c6 最低。

### 3.4 落子策略

对排序后的 **top-5 候选**（c1 至 c5）基于 Q_search 做 softmax 采样。

#### Scale Normalization

为保证不同局面的采样行为一致，先对 Q_search 做线性拉伸：

```
Q_min = min(Q_search(c1), ..., Q_search(c5))
Q_max = max(Q_search(c1), ..., Q_search(c5))
Q_norm(ci) = (Q_search(ci) - Q_min) / (Q_max - Q_min + ε)    # 映射到 [0, 1]
```

其中 ε 为小常数（如 1e-6），防止除零。

#### Softmax 采样

```
prob(ci) ∝ exp(Q_norm(ci) / τ)    for i in {1, 2, 3, 4, 5}
```

其中 τ 为温度参数，控制随机性程度：
- τ 较小：分布尖锐，倾向选择 c1
- τ 较大：分布平坦，增加选择次优动作的概率

**排除 c6**：c6 可能是随机候选中的差招，不参与采样。

## 4. 训练目标

### 4.1 Policy Loss：排序损失

目标：使 policy logits 的顺序与搜索排序一致。

记模型对动作 x 的 logit 为 `L(x)`。

#### 4.1.1 候选内部排序约束（Ranking-Inside）

仅约束前 4 名（c1 至 c4）的相对顺序，使用动态 margin 处理平局情况：

```
margin(i, j) = min(m_rank, Q_norm(ci) - Q_norm(cj))

RankingInsideLoss = ReLU(L(c2) - L(c1) + margin(1, 2))
                  + ReLU(L(c3) - L(c2) + margin(2, 3))
                  + ReLU(L(c4) - L(c3) + margin(3, 4))
```

**动态 margin 的作用**：
- 当 Q 值差距大于 m_rank 时，使用完整的 m_rank 作为 margin
- 当 Q 值差距较小时，margin 与实际差距成比例（避免过度约束）
- 当 Q 值相等（平局）时，margin = 0，仅惩罚逆序（L(cj) > L(ci)），不强制分离

#### 4.1.2 非候选分离约束（Separation-Outside）

防止非候选动作的 logit 反超进入 top-k。对所有非候选动作 n（不在 C(s) 中的合法动作）：

```
SeparationOutsideLoss = mean over n of ReLU(L(n) - L(c4) + m_sep)
```

边界为 c4（第 4 名），使用 **mean** 而非 sum 以平衡 scale。

#### 4.1.3 Policy 总损失

```
PolicyLoss = RankingInsideLoss + α * SeparationOutsideLoss
```

### 4.2 Value Loss：搜索备份值监督

```
V_target(s) = max_{a in C(s)} Q_search(s, a)
ValueLoss = (V_pred(s) - V_target(s))^2
```

Value 语义保持为「当前行动方视角」的估值，与现有 canonical 观测一致。

### 4.3 总损失

```
TotalLoss = PolicyLoss + λ_v * ValueLoss
```

**注意**：不使用 entropy loss。

## 5. 训练阶段与冻结策略

采用渐进式解冻：

| 阶段 | Update 区间 | 可训练参数 |
|------|-------------|-----------|
| 阶段 1 | [0, N) | 仅 policy head + value head |
| 阶段 2 | [N, N+M) | + trunk 最后 1 个 block |
| 阶段 3 | [N+M, N+2M) | + trunk 倒数第 2 个 block |
| ... | ... | 每 M 个 update 解冻 1 个 block |
| 阶段 k | 所有 trunk blocks 解冻后 | + stem（全网络可训练） |

解冻顺序：policy head + value head → trunk blocks（从后向前）→ stem

## 6. 超参数

| 参数 | 说明 | 值 |
|------|------|-----|
| m_rank | 候选内部排序 margin（最大值，实际 margin 为 min(m_rank, Q_norm 差值)） | 0.15 |
| m_sep | 非候选分离 margin | 0.15 |
| α | SeparationOutsideLoss 权重 | 1.0 |
| λ_v | ValueLoss 权重 | 1.0 |
| τ | 落子采样温度 | 0.5 |
| ε | Q normalization 防除零常数 | 1e-6 |
| N | heads-only 阶段的 update 数 | 2048 |
| M | 每个 trunk block 解冻间隔的 update 数 | 128 |

## 7. Self-Play 流程

```
for each game:
    state = initial_state()
    while not terminal(state):
        player = current_player(state)
        model = player_model[player]  # current or opponent

        # 生成候选集
        candidates = generate_candidates(state, model)  # 6 个

        # Negamax 搜索
        Q_search = negamax(state, candidates, depth=3)

        # 排序
        sorted_candidates = sort_by_Q_descending(candidates, Q_search)
        c1, c2, c3, c4, c5, c6 = sorted_candidates

        # 记录训练样本（仅记录 current model 的回合）
        if model == current_model:
            record_sample(
                obs=state.observation(),
                sorted_candidates=[c1, c2, c3, c4],
                all_candidates=candidates,
                V_target=max(Q_search.values())
            )

        # 落子：对 top-5 做 softmax 采样（带 scale normalization）
        top5_Q = [Q_search[c] for c in [c1, c2, c3, c4, c5]]
        Q_min, Q_max = min(top5_Q), max(top5_Q)
        Q_norm = [(q - Q_min) / (Q_max - Q_min + ε) for q in top5_Q]
        probs = softmax([q / τ for q in Q_norm])
        action = sample([c1, c2, c3, c4, c5], probs)
        state = state.apply(action)
```

## 8. 验证指标

训练过程中跟踪以下指标以评估训练质量：

| 指标 | 说明 |
|------|------|
| Top-1 Ranking Accuracy | c1 是否具有最高 logit 的比例 |
| Top-3 Ranking Accuracy | c1, c2, c3 是否为 logit 最高的 3 个动作的比例 |
| Value MSE | V_pred 与 V_target 的均方误差 |
| Win Rate | 定期对弈评估胜率 |

## 9. 与现有系统的集成

- **对手池**：沿用现有采样策略
- **观测表示**：沿用 canonical 表示（channel 0 = 当前方，channel 1 = 对手，channel 2 = mask）
- **Checkpoint 格式**：与现有格式兼容
- **数据增强**：沿用 8-fold 对称增强（对候选动作同步变换）
