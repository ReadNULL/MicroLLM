"""LoRA层

在所有层注入可训练的低秩矩阵A 和 B：
    output = W·x + (α/r)·ΔW·x = W·x + (α/r)·B·A·x

其中增量矩阵ΔW = B·A, B(d_out, r), A(r, d_in)

原始参数矩阵W会被冻结，只有矩阵A 和 B会被更新。
"""
import math

import torch
import torch.nn as nn

from .transformer import Linear, Identity


class LoRA(nn.Module):
    def __init__(
            self,
            original: Linear,
            r: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"Rank r must be positive, got {r}")
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")

        out_features, in_features = original.weight.shape
        if r > min(out_features, in_features):
            raise ValueError(f"Rank r ({r}) cannot exceed dimensions ({out_features}, {in_features})")

        self.original = original
        self.original.requires_grad_(False) # 冻结原始参数矩阵W
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()

        out_features, in_features = original.weight.shape
        device = self.original.weight.device
        dtype = original.weight.dtype

        # B(d_out, r), A(r, d_in)
        self.B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype), requires_grad=True)
        self.A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype), requires_grad=True)
        # 矩阵A使用kaiming正则初始化，B使用0初始化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = x @ self.original.weight.T
        if self._merged:
            return original_out
        # (α/r)·ΔW·x = (α/r)·B·A·x
        lora_out = (self.dropout(x) @ self.A.T) @ self.B.T * self.scaling
        return original_out + lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """
            将lora权重参数合并到原始权重参数
        """
        if self._merged:
            return
        delta = (self.B @ self.A) * self.scaling
        self.original.weight.add_(delta)
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """
            将lora权重从原始权重中拆解
        """
        if not self._merged:
            return

        delta = (self.B @ self.A) * self.scaling
        self.original.weight.sub_(delta)
        self._merged = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def get_delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scaling

# ---- apply LoRA to a TransformerLM ----------------------------------------

_DEFAULT_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj"}

def _replace_module(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

def apply_lora_to_model(model, r=8, alpha=16.0, dropout=0.0, target_names=None):
    # 1. 全局冻结
    for p in model.parameters():
        p.requires_grad_(False)

    if target_names is None:
        target_names = _DEFAULT_TARGETS

    target_set = set(target_names)

    # 收集所有需要嵌入的项
    replacements = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_set and isinstance(module, Linear):
            replacements.append((name, LoRA(module, r=r, alpha=alpha, dropout=dropout)))

    # 执行嵌入
    for name, lora_layer in replacements:
        _replace_module(model, name, lora_layer)

def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRA):
            parameters.append(module.A)
            parameters.append(module.B)
    return parameters

def get_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state_dicts: dict[str, torch.Tensor] = {}
    for name, module in module.named_modules():
        if isinstance(module, LoRA):
            state_dicts[f"{name}.A"] = module.A.data.cpu()
            state_dicts[f"{name}.B"] = module.B.data.cpu()
    return state_dicts

def merge_all_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRA):
            module.merge()

def unmerge_all_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRA):
            module.unmerge()

def print_trainable_weights(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total * 100.0 if total > 0 else 0
    print(f"Total params: {total:,} | Trainable (LoRA): {trainable:,} ({ratio:.2f}%)")
    # LoRA参数统计
    lora_params = sum(p.numel() for m in model.modules()
                      if isinstance(m, LoRA) for p in m.parameters() if p is not None)
    if lora_params:
        print(f"LoRA params: {lora_params:,} ({lora_params/total*100:.2f}% of total)")
