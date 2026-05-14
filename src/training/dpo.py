from __future__ import annotations

import torch


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Direct Preference Optimization loss.

    Reference:
        Direct Preference Optimization: Your Language Model is Secretly a Reward Model
        (Rafailov et al., 2023)

    The DPO loss is:
        L_DPO = -log sigmoid(beta * (log(pi_w(y_c|x)) - log(pi_ref(y_c|x))
                                        - (log(pi_w(y_r|x)) - log(pi_ref(y_r|x)))))

    Args:
        policy_chosen_logps: Log-probabilities of chosen responses under the policy model.
            Shape: (batch_size,)
        policy_rejected_logps: Log-probabilities of rejected responses under the policy model.
            Shape: (batch_size,)
        reference_chosen_logps: Log-probabilities of chosen responses under the reference model.
            Shape: (batch_size,)
        reference_rejected_logps: Log-probabilities of rejected responses under the reference model.
            Shape: (batch_size,)
        beta: Temperature parameter controlling the divergence from the reference model.
            Typical values: 0.1 ~ 0.5.

    Returns:
        loss: Scalar DPO loss.
        chosen_rewards: Per-sample chosen rewards (implicit reward model).
        rejected_rewards: Per-sample rejected rewards (implicit reward model).
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)

    loss = -torch.nn.functional.logsigmoid(logits).mean()

    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return loss, chosen_rewards, rejected_rewards


def compute_token_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample log-probabilities from logits given token-level labels and a mask.

    This function computes the sum of log-probabilities for each token position
    where the mask is 1, then sums over the sequence dimension to get one scalar
    per sample in the batch.

    Args:
        logits: Predicted logits. Shape: (batch_size, seq_len, vocab_size)
        labels: Target token IDs. Shape: (batch_size, seq_len)
        loss_mask: Mask indicating which tokens to include. Shape: (batch_size, seq_len)
            1 means include, 0 means ignore.

    Returns:
        logps: Per-sample summed log-probabilities. Shape: (batch_size,)
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels must agree on batch/time dimensions")
    if labels.shape != loss_mask.shape:
        raise ValueError("labels and loss_mask must have the same shape")

    log_probs = logits.log_softmax(dim=-1)

    safe_labels = labels.clamp_min(0)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    mask = loss_mask.to(token_logps.dtype)
    per_sample_logps = (token_logps * mask).sum(dim=-1)

    return per_sample_logps
