import math

import torch.nn as nn
import torch
from einops import rearrange


class Linear(nn.Module):
    """
        自定义的线性层，使用 Xavier 截断正则化初始化权重参数
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        # 线性层权重，作为可训练参数放到nn.Parameter中
        self.weight = nn.Parameter(torch.empty(in_features, out_features, **kwargs))
        self.bias = nn.Parameter(torch.zeros(out_features, **kwargs))
        # 使用截断正则化初始化权重参数，std做为标准差σ，将所有参数限制在[-3σ, 3σ]区间以内
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        # 前向传播，等价于x @ self.weight.T + bias
        return torch.einsum("... i, o i -> ... o", x, self.weight) + self.bias

class Embedding(nn.Module):
    """
        自定义的嵌入层，使用截断正则化初始化权重
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **kwargs))

        # std做为标准差σ默认为1.0，将所有参数限制在[-3σ, 3σ]区间以内
        std = math.sqrt(1.0 / embedding_dim)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

        # 设置默认填充词的索引和词向量
        self.padding_idx = padding_idx
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
            前向传播：根据token_ids = [1, 2...]
            直接从权重表[num_embeddings, embedding_dim]中
            取出第[1, 2...]行作为对应的词向量返回
        """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
        自定义的RMS正则化层，先将参数全部投影到更高精度float32计算，再投影回原精度类型
        ReLU等激活函数已将特征分布近似零均值化，因此可省略均值中心化步骤
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float32 = x.to(torch.float32)
        # 归一化计算公式：x' = x / RMS(x) * γ, RMS(x) = sqrt((x²).mean + eps)
        rms = torch.rsqrt(x_float32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_float32 * rms) * self.weight.to(torch.float32)).to(in_dtype)

class SwiGLU(nn.Module):
    """
        自定义SwiGLU激活函数，计算公式：
        Swish(x) = x * σ(x)
        SwiGLU(x) = W2( Swish(W1(x)) ⊙ W3(x) )
        其中σ为sigmoid函数
        W1: 门控路径，经过 SiLU 激活, 维度 d_model * d_ff
        W2: 主路径，保持原始变换 维度 d_ff * d_model
        W3: 输出投影，将结果映射回 d_model 维度 d_model * d_ff
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        swish_x = w1_out * torch.sigmoid(w1_out)
        return self.w2(swish_x * self.w3(x))

class RoPE(nn.Module):
    """
        自定义RoPE类，
        对于位置 m 和维度 i：
        q_m = R_{Θ,m} · q
        k_n = R_{Θ,n} · k
        其中，旋转矩阵是分块对角矩阵，每个块是 2D 旋转
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 判定d_k 为偶数
        if d_k % 2 != 0:
            raise ValueError("d_k must be divisible by 2 for RoPE")
        # 计算旋转频率，公式：θ_i = θ^{-2i/d_k}, 其中 i = 0, 1, ..., d_k/2 - 1
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-indices / d_k)
        # 计算所有位置的旋转角度
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # angles 矩阵：
        # 位置 0: [0, 0, 0, 0]
        # 位置 1: [θ₁, θ₂, θ₃, θ₄]
        # 位置 2: [2θ₁, 2θ₂, 2θ₃, 2θ₄]
        # ...
        angeles = torch.outer(positions, inv_freq)

        # 缓存sin和cos表
        self.register_buffer('cos_cached', torch.cos(angeles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angeles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 获取对应位置的cos和sin值
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # 处理维度匹配
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # 将x的最后一维分成两半：[..., seq, d_k] -> [..., seq, d_k/2, 2]
        # 例如：
        # 输入 x: [2, 3, 8]
        # 输出 x_pairs: [2, 3, 4, 2]
        # 其中最后一维的两个值是需要旋转的一对
        x_pairs = rearrange(x, "... seq (pair two) -> ... seq pair two", pair=2)
        # 应用旋转变换
        cos = cos.unsqueeze(-1)  # [..., seq, d_k/2, 1]
        sin = sin.unsqueeze(-1)  # [..., seq, d_k/2, 1]
        x_even = x_pairs[..., 0:1]  # 取偶数索引
        x_odd = x_pairs[..., 1:2]  # 取奇数索引
        # 旋转公式：
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_even * sin + x_odd * cos
        rotated = torch.cat((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rearrange(rotated, "... seq pair two -> ... seq (pair two)")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        softmax函数实现：softmax(x) = exp(x_i - max({x_i})) / ∑(i = 1...n) exp(x_i - max({x_i}))
        输出与x同维度的概率分布
    """
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / (exp_shifted.sum(dim=dim, keepdim=True) + 1e-8)

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None) -> torch.Tensor:
    """
        注意力分数计算：Attention(Q, K, V) = softmax(Q * K ^ T / sqrt(d_k)) * V
    """
    d_k = q.shape[-1]
    # 使用float32计算以确保fp16推理的精度
    attn_dtype = v.dtype
    scores = torch.einsum(
        "... q d, ... k d -> ... q k",
        q.to(torch.float32),
        k.to(torch.float32)
    ) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(attn_dtype).min)
    scores = softmax(scores, dim=-1).to(attn_dtype)
    return torch.einsum(
        "... q k, ... k d -> ... q d",
        scores,
        v.to(attn_dtype)
    )

class KVCache:
    """
        自定义的KVCache类
    """
    def __init__(self, num_layers: int):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def reset(self):
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None

    def update(self, layer_idx, k, v):
        self.k[layer_idx] = k
        self.v[layer_idx] = v

    def append(self, layer_idx, k_new: torch.Tensor, v_new: torch.Tensor):
        if k_new and v_new:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=-1)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=-1)

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]


class MultiHeadAttention(nn.Module):
    def __init__(self,
            d_model: int,
            num_heads: int,
            max_seq_len: int | None = None,
            theta: float = None,
            device=None,
            dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # qkv聚合计算
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x, token_positions = None, past_k = None, past_v = None, use_cache = False):
        seq_len = x.shape[-2]
        leading_shape = x.shape[:-2]
        # Q, K, V投影 + 拆头
        # 性能优化，qkv放在一起计算最后拆分
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d',
                            three=3, h=self.num_heads, d=self.d_k)

        # RoPE位置编码
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                # 扩展维度以匹配 q 的维度
                for _ in range(q.ndim - token_positions.ndim):
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.expand(*q.shape[:-1])
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # KV Cache分支 & 标准分支
        if use_cache:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)
            attn_out = scaled_dot_product_attention(q, k, v, mask=None)
            new_k, new_v = k, v
        else:
            casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
            attn_out = scaled_dot_product_attention(q, k, v, mask=casual_mask)
            new_k, new_v = None, None

        attn_out = rearrange(attn_out, "... head seq d -> ... seq (head d)")
        out = self.out_proj(attn_out)
        return out, new_k, new_v

class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class TransformerBlock(nn.Module):
    """
        Transformer块
    """
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, max_seq_len: int, theta: float,
                 use_rms_norm = True, norm_mode = 'pre',
                 ffn_type = 'swiglu', device=None, dtype=None
                 ):
        super().__init__()
        self.norm_mode = norm_mode
        norm_cls = lambda: RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,
                                       theta=theta, device=device, dtype=dtype)
        self.norm_layer1 = norm_cls()
        self.norm_layer2 = norm_cls()
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions=None, past_k=None, past_v=None, use_cache=False):
        if self.norm_mode == "post":
            raise ValueError(f"Normalization mode {self.norm_mode} is not supported. Use 'pre' instead.")

        h = self.norm_layer1(x)
        attn_out, new_k, new_v = self.attn(h, token_positions=token_positions, past_k=past_k, past_v=past_v, use_cache=use_cache)
        x = x + attn_out # 残差连接1
        x = x + self.ffn(self.norm_layer2(x)) # 正则化 + 残差连接2
        return x, new_k, new_v

class TransformerLM(nn.Module):
    """
        Transformer解码器
    """
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            use_rms_norm: bool = True,
            norm_mode: str = "pre",
            ffn_type: str = "swiglu",
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(num_embeddings=vocab_size,
                                         embedding_dim=d_model,
                                         padding_idx=-1,
                                         device=device,
                                         dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.final_layer = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.output_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, kv_cache=None, use_cache=False, start_pos=0):
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"seq_len overflow: max sequence length must be less than {self.context_length}")

        leading_shape = token_ids.shape[:-1]
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=token_ids.device)
        token_positions = (token_positions.view(*([1] * len(leading_shape)), seq_len)
                           .expand(*leading_shape, seq_len))
        x = self.token_embedding(token_ids)
        if use_cache and kv_cache is None:
            kv_cache = KVCache(len(self.layers))

        for layer_idx, layer in enumerate(self.layers):
            if use_cache:
                x, new_k, new_v = layer(
                    x, token_positions=token_positions,
                    past_k=kv_cache.k[layer_idx], past_v=kv_cache.v[layer_idx],
                    use_cache=True
                )
                kv_cache.k[layer_idx] = new_k
                kv_cache.v[layer_idx] = new_v
            else:
                x, _, _ = layer(x, token_positions=token_positions)

        x = self.final_layer(x)
        out = self.output_layer(x)

        return out, kv_cache

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_p=1.0):
        """
        生成/预测下一个词
        :param prompt_ids: 提示词的token_id列表
        :param max_new_tokens: 支持生成最大的token数
        :param eos_token_id: 结束符token_id
        :param temperature: 温度参数
        :param top_p: top_p概率和
        :return:
        """
        self.eval()
        generated_token_ids = prompt_ids.clone()
        kv_cache = KVCache(len(self.layers))

        # 阶段1：Prefill，默认开启KVCache
        out, kv_cache = self.forward(
            token_ids=prompt_ids,
            kv_cache=kv_cache,
            use_cache=True,
            start_pos=0
        )
        out = out[:, -1, :]

        # 阶段2：Decode Loop
        for _ in range(max_new_tokens):
            if temperature != 1.0:
                out = out / temperature
            if top_p < 1.0:
                out = self._top_p_filter(out, top_p=top_p)
            probs = softmax(out, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_token_ids = torch.cat((generated_token_ids, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            cur_pos = generated_token_ids.shape[1] - 1
            out, kv_cache = self.forward(
                next_token, kv_cache=kv_cache,
                use_cache=True, start_pos=cur_pos
            )
            out = out[:, -1, :]     # 每次只取最后一个位置

        return generated_token_ids

    def _top_p_filter(self, out, top_p):
        sorted_out, sorted_indices = torch.sort(out, descending=True, dim=-1)
        total_prob = torch.cumsum(softmax(sorted_out, dim=-1), dim=-1)
        sorted_indices_to_remove = total_prob > top_p
        # "右移一位"技巧：保留第一个越界 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        out = out.masked_fill(indices_to_remove, float("-inf"))
        return out

















