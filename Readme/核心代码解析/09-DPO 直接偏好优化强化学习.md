---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P0
file: src/training/dpo.py + dpo_dataset.py + scripts/train_dpo.py
---

# DPO — 直接偏好优化强化学习训练

> DPO（Direct Preference Optimization）是从 SFT 到对齐（Alignment）的关键一步。与 RLHF 相比，DPO 不需要训练独立的 reward model，而是将偏好数据直接映射为隐式奖励信号，通过单一训练循环优化模型使其更偏好 chosen 回答而非 rejected 回答。这是项目从"能回答"升级到"回答得更好"的核心环节。

---

## 1. 为什么需要 DPO

SFT 阶段让模型学会了按照对话格式回答问题，但它无法区分"好回答"和"差回答"——只要格式正确、语言通顺，SFT 都会给予相同的训练信号。

**SFT 的局限：** 模型不知道用户的偏好。两个回答都合法，但一个更准确、一个有幻觉；SFT 无法区分。

**DPO 的解法：** 给定同一 prompt 的 chosen（优选）和 rejected（拒绝）两个回答，通过优化使模型更倾向于生成 chosen 内容，同时用 KL 散度约束防止偏离参考模型太远。

### DPO vs RLHF 对比

| 维度 | RLHF | DPO |
|------|------|-----|
| Reward Model | 需要独立训练 | 隐式（从策略模型推导） |
| 训练阶段 | 3 阶段（SFT → RM → RL） | 2 阶段（SFT → DPO） |
| 训练复杂度 | 高（PPO 需要 value network） | 低（直接计算 loss） |
| 内存开销 | 大（多个模型 + 采样） | 小（policy + reference） |
| 效果 | 经典方法 | 理论等价，实践中相当或更好 |

---

## 2. DPO 数据协议

### 2.1 JSONL 数据格式

```jsonl
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "什么是机器学习？"}
  ],
  "chosen": [{"role": "assistant", "content": "机器学习是让计算机从数据中学习的技术..."}],
  "rejected": [{"role": "assistant", "content": "机器学习就是电脑自己写代码。"}]
}
```

**设计决策：**
- **prompt 独立于 chosen/rejected**：避免 prompt 与回答耦合，便于构建对比样本
- **chosen/rejected 使用相同的 sft.py 协议**：复用 `normalize_conversations` → `render_chat_prompt` → `build_loss_labels` 管线
- **只有 assistant 区间参与 log-prob 计算**：与 SFT 保持一致

### 2.2 DPODataset 处理管线

```
JSONL 样本
    │
    ▼ 读取 prompt/chosen/rejected
    │
    ▼ normalize_conversations(prompt + chosen)
    ▼ normalize_conversations(prompt + rejected)
    │
    ▼ _encode_conversations() — 复用 sft.py 管线
    │   render_chat_prompt → encode → build_loss_labels
    │
    ▼ 返回 dict:
    {
        "chosen_input_ids": [B, L],
        "chosen_labels": [B, L],           # assistant-only labels
        "chosen_loss_masks": [B, L],       # 从 labels 推导
        "rejected_input_ids": [B, L],
        "rejected_labels": [B, L],
        "rejected_loss_masks": [B, L],
    }
```

**关键：** loss_masks 从 labels 自动推导（`label != -100` 的位置即为 1），无需重新搜索边界。

---

## 3. ★ DPO 损失函数 — 核心算法

### 3.1 理论公式

DPO 的核心创新是将奖励建模和策略优化统一在一个框架中：

```
L_DPO(π_θ; π_ref) = -E_{(x,y_c,y_r)~D}[log σ(β log(π_θ(y_c|x)/π_ref(y_c|x)) - β log(π_θ(y_r|x)/π_ref(y_r|x)))]
```

**直观解释：**
- `log(π_θ(y|x) / π_ref(y|x))` 是隐式奖励（implicit reward）
- DPO 优化目标是拉大 chosen 和 rejected 之间的奖励差距
- β 控制与参考模型的偏离程度（KL 惩罚强度）

### 3.2 代码实现

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,      # [B]
    policy_rejected_logps: torch.Tensor,    # [B]
    reference_chosen_logps: torch.Tensor,   # [B]
    reference_rejected_logps: torch.Tensor, # [B]
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    
    logits = beta * (chosen_logratios - rejected_logratios)
    
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return loss, chosen_rewards, rejected_rewards
```

**关键设计点：**

| 设计 | 原因 |
|------|------|
| 用 `logsigmoid` 而非 `-log sigmoid` | 数值稳定性更好，PyTorch 原生实现 |
| rewards 用 `.detach()` | 奖励只是监控指标，不参与梯度传播 |
| 返回 3 个值 | loss 用于优化，rewards 用于评估对齐质量 |

### 3.3 compute_token_logps — 序列级 log 概率

```python
def compute_token_logps(
    logits: torch.Tensor,     # [B, L, V]
    labels: torch.Tensor,     # [B, L]
    loss_mask: torch.Tensor,  # [B, L]
) -> torch.Tensor:            # [B]
    
    log_probs = logits.log_softmax(dim=-1)
    
    safe_labels = labels.clamp_min(0)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    
    mask = loss_mask.to(token_logps.dtype)
    per_sample_logps = (token_logps * mask).sum(dim=-1)
    
    return per_sample_logps
```

**为什么这样计算：**
1. `log_softmax` 比 `log(softmax(x))` 数值稳定
2. `clamp_min(0)` 防止 labels 中的 -100 导致 gather 索引越界（mask 会过滤掉）
3. 对序列长度维度求和得到样本级 log-prob，与 DPO 论文一致

---

## 4. 训练循环架构

### 4.1 双模型架构

```
┌─────────────────────────────────────────────────┐
│                  DPO 训练循环                     │
│                                                  │
│  策略模型 (policy)                               │
│  ├─ 可训练（梯度更新）                           │
│  ├─ 从 SFT checkpoint 初始化                     │
│  └─ 可选 LoRA 高效微调                           │
│                                                  │
│  参考模型 (reference)                            │
│  ├─ 冻结（requires_grad=False）                  │
│  ├─ 与策略模型同架构、同权重初始化               │
│  └─ 提供基准分布（KL 散度的锚点）                │
└─────────────────────────────────────────────────┘
```

**为什么需要两个模型：** DPO loss 需要同时计算 policy 和 reference 的 log-prob。reference 模型在训练过程中保持不变，提供 KL 惩罚的基准。

### 4.2 训练步骤详解

```python
# 1. 前向传播（策略模型）
chosen_logits = model(chosen_input_ids)
rejected_logits = model(rejected_input_ids)

# 2. 参考模型前向（无梯度）
with torch.no_grad():
    ref_chosen_logits = ref_model(chosen_input_ids)
    ref_rejected_logits = ref_model(rejected_input_ids)

# 3. 计算 log-probabilities
policy_chosen_logps = compute_token_logps(chosen_logits, chosen_labels, chosen_loss_masks)
policy_rejected_logps = compute_token_logps(rejected_logits, rejected_labels, rejected_loss_masks)
ref_chosen_logps = compute_token_logps(ref_chosen_logits, chosen_labels, chosen_loss_masks)
ref_rejected_logps = compute_token_logps(ref_rejected_logits, rejected_labels, rejected_loss_masks)

# 4. 计算 DPO loss
dpo_loss_val, chosen_rewards, rejected_rewards = dpo_loss(
    policy_chosen_logps, policy_rejected_logps,
    ref_chosen_logps, ref_rejected_logps,
    args.dpo_beta
)

# 5. 可选：混合 SFT 辅助损失
loss = dpo_loss_val
if args.sft_weight > 0.0:
    sft_loss = masked_cross_entropy(chosen_logits, chosen_labels, chosen_loss_masks)
    loss = dpo_loss_val + args.sft_weight * sft_loss

# 6. 反向传播 + 优化
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

### 4.3 SFT 辅助损失

**为什么需要：** 纯 DPO 训练可能导致模型忘记基本的语言流畅性（language drift）。`--sft-weight` 参数允许在 DPO loss 基础上叠加一小部分 SFT loss（通常 0.1~0.2），保持生成质量。

```
总损失 = DPO_loss + sft_weight × SFT_loss(chosen)
```

### 4.4 评估与监控

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `dpo_loss` | 偏好对齐损失 | 下降 |
| `val_loss` | 验证集 DPO loss | 下降 |
| `chosen_rewards` | chosen 样本隐式奖励 | 上升 |
| `rejected_rewards` | rejected 样本隐式奖励 | 下降或持平 |
| `reward_margin` | chosen - rejected | 扩大（>0） |

---

## 5. 配置与使用

### 5.1 配置文件

```json
{
  "training": {
    "batch_size": 1,
    "max_steps": 100,
    "init_checkpoint": "outputs/sft_smoke/ckpt_final.pt",
    "out_dir": "outputs/dpo"
  },
  "optimizer": {
    "lr": 5e-6
  },
  "dpo": {
    "beta": 0.1,
    "sft_weight": 0.0
  }
}
```

**关键超参：**
- `beta`（0.1~0.5）：KL 惩罚强度。越大表示越不敢偏离 reference 模型
- `lr`（1e-6~1e-5）：DPO 学习率通常比 SFT 小一个数量级
- `sft_weight`（0.0~0.2）：SFT 辅助损失权重

### 5.2 使用方式

```bash
# 基本用法
python -m scripts.train_dpo --config configs/dpo_smoke.json

# 带 LoRA 高效微调
python -m scripts.train_dpo --config configs/dpo_smoke.json --use-lora --lora-r 8

# 混合 SFT 辅助损失
python -m scripts.train_dpo --config configs/dpo_smoke.json --sft-weight 0.1

# 命令行覆盖配置
python -m scripts.train_dpo --config configs/dpo_smoke.json --dpo-beta 0.2 --lr 1e-5
```

---

## 6. 与项目其他组件的关系

```
SFT 模型 (ckpt_final.pt)
    │
    ├──→ train_dpo.py ──→ DPO 模型 (ckpt_final.pt)
    │                          │
    │                          ├──→ generate_text.py（推理）
    │                          ├──→ chat.py（多轮对话）
    │                          └──→ run_instructie_eval.py（评测）
    │
    └──→ train_dpo.py 的 reference_model
         （冻结，仅用于 KL 惩罚）
```

**DPO 是 SFT 的下游阶段：** 必须先用 SFT 模型初始化策略模型，不能从 random 或 pretrain 权重直接开始 DPO。

---

## 7. 面试高频追问

| 问题 | 回答要点 |
|------|----------|
| DPO 相比 RLHF 有什么优势？ | 不需要独立训练 reward model，不需要 PPO，实现更简单且效果相当 |
| DPO 的 KL 惩罚是怎么实现的？ | 通过 policy/reference log-prob ratio 隐式实现，β 控制惩罚强度 |
| β 参数怎么调？ | 太小容易过拟合偏好数据，太大则学习太慢。通常 0.1~0.5 |
| 为什么需要 reference 模型？ | 提供基准分布，防止策略模型过度优化偏好数据而忘记语言能力 |
| DPO 数据集怎么构造？ | 同一 prompt + 两个回答（好/差），可以用模型生成 + 人工标注 |
| 训练时怎么监控对齐效果？ | 观察 chosen_rewards 上升、rejected_rewards 下降、reward_margin 扩大 |

---

## 相关记录

- [[03-sft.py SFT 数据协议]] — DPO 数据集复用的对话协议和 loss label 构造逻辑
- [[01-transformer.py 模型主干]] — DPO 训练使用的模型架构
- [[02-lora.py LoRA 参数高效微调]] — DPO 支持 LoRA 高效微调
- [[04-data_loader.py 与 loss.py]] — 对比 pretrain 的简单 cross-entropy loss
- [[06-项目复盘与总结]] — DPO 在项目整体链路中的定位
