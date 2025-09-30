from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset2 import save_emb
from pathlib import Path
import numpy as np

import math

import torch


class PScan(torch.autograd.Function):

    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2 ** k - 1: L: 2 ** k]
            Xa = X[:, :, 2 ** k - 1: L: 2 ** k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


def selective_scan(x, delta, A, B, C, D):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (
            hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
            hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class MambaBlock(nn.Module):
    def __init__(self, dim, dt_rank, dim_inner, d_state, d_conv=4):
        super().__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.d_conv = d_conv

        # FIXED: Single input projection that splits into two paths
        self.in_proj = nn.Linear(dim, dim_inner * 2, bias=False)

        # FIXED: Proper depthwise causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            kernel_size=d_conv,
            groups=dim_inner,  # Depthwise
            padding=d_conv - 1,  # Causal padding
            bias=True
        )

        self.activation = nn.SiLU()
        self.ssm = SSM(in_features=dim_inner, dt_rank=dt_rank, dim_inner=dim_inner, d_state=d_state)
        self.out_proj = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        batch, seqlen, dim = x.shape

        # FIXED: Split input projection into two paths
        xz = self.in_proj(x)  # (B, L, 2*dim_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, dim_inner)

        x = x.transpose(1, 2)  # (B, dim_inner, L)
        x = self.conv1d(x)[..., :seqlen]  # Slice for causal
        x = x.transpose(1, 2)  # (B, L, dim_inner)
        x = self.activation(x)

        x = self.ssm(x)  # (B, L, dim_inner)

        x = x * self.activation(z)

        out = self.out_proj(x)  # (B, L, dim)

        return out


class MultiHeadMamba(nn.Module):
    def __init__(self,
                 dim,
                 dt_rank,
                 dim_inner,
                 d_state,
                 args
                 ):
        super().__init__()
        self.args = args
        self.head_dim = dim // args.hydra_num_heads
        dim_inner_head = 4 * self.head_dim  # Expansion factor 4 per head
        d_state_head = d_state  # Fixed state size per head
        self.mamba = nn.ModuleList([MambaBlock(
            self.head_dim, dt_rank, dim_inner_head, d_state_head
        ) for _ in range(args.hydra_num_heads)])
        self.activation = nn.SiLU()
        self.linear_layer1 = nn.Linear(dim, dim)
        self.linear_layer2 = nn.Linear(dim, dim)

    def get_rotary_matrix(self, seq_len, head_dim, device, base=10000):
        """
        Compute the rotary matrix for RoPE.

        Args:
            seq_len: Sequence length
            head_dim: Dimension of each attention head
            device: Device to place the tensors
            base: Base frequency for theta (default: 10000)

        Returns:
            cos_cached: Cosine of the rotation angles
            sin_cached: Sine of the rotation angles
        """
        # Ensure head_dim is even
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

        # Compute theta values
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))

        # Create position indices [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Compute angles: outer product of positions and theta
        angles = positions[:, None] * theta[None, :]  # Shape: (seq_len, head_dim/2)

        # Compute cos and sin of angles
        cos_cached = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim/2)
        sin_cached = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1, head_dim/2)

        return cos_cached, sin_cached

    def apply_rotary_emb(self, x, cos_cached, sin_cached):
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            cos_cached: Cosine of rotation angles, shape (1, seq_len, 1, head_dim/2)
            sin_cached: Sine of rotation angles, shape (1, seq_len, 1, head_dim/2)

        Returns:
            Rotated tensor of the same shape as input
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        x = x.view(batch_size, seq_len, num_heads, head_dim // 2, 2)

        # Split into x1, x2 (first and second dimensions of each pair)
        x1, x2 = x[..., 0], x[..., 1]

        # Apply rotation: [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        rotated_x1 = x1 * cos_cached - x2 * sin_cached
        rotated_x2 = x1 * sin_cached + x2 * cos_cached

        # Combine rotated dimensions
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).view(batch_size, seq_len, num_heads, head_dim)

        return rotated_x

    def forward(self, x, x_org, padding_mask):
        batch_size, seq_len, dim = x.shape
        if padding_mask is not None:
            h1 = x * padding_mask
        else:
            h1 = x
        # path 1
        x1 = self.linear_layer1(h1)
        x2 = x1
        x1 = self.activation(x1)
        x1 = x1.view(batch_size, seq_len, self.args.hydra_num_heads, self.head_dim)
        x_split = torch.chunk(x1, self.args.hydra_num_heads, dim=2)
        x_split = [X.squeeze(-2) for X in x_split]
        x1_outputs = []
        for i, mamba in enumerate(self.mamba):
            head_input = x_split[i].squeeze(2)
            head_output = mamba(head_input)
            x1_outputs.append(head_output)

        y1 = torch.cat(x1_outputs, dim=-1)

        # path2
        x2 = self.activation(x2)
        x2 = x2.view(batch_size, seq_len, self.args.hydra_num_heads, self.head_dim)
        cos_cached, sin_cached = self.get_rotary_matrix(seq_len, self.head_dim, self.args.device)
        y2 = self.apply_rotary_emb(x2, cos_cached, sin_cached)

        y2 = y2.view(batch_size, seq_len, dim)
        out = (y1 * y2) / (self.args.hydra_num_heads ** 0.5)
        out = x_org + self.linear_layer2(out)
        return out


class HydraBlockFirst(nn.Module):
    def __init__(self,
                 dim,
                 dt_rank,
                 dim_inner,
                 d_state,
                 args,
                 ):
        super().__init__()
        self.multiheadMamba = MultiHeadMamba(dim, dt_rank, dim_inner, d_state, args)
        self.gate = nn.Linear(dim, dim, bias=False)  # W_gate
        self.up = nn.Linear(dim, dim, bias=False)  # W_up
        self.down = nn.Linear(dim, dim, bias=False)  # W_down
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=args.hydra_dropout_rate)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, padding_mask):
        norm_x = self.norm1(self.dropout(x))
        mamba_x = norm_x + self.multiheadMamba(norm_x, x, padding_mask)

        # FFN with pre-norm
        norm2_x = self.norm2(mamba_x)
        gate_x = self.activation(self.gate(norm2_x))
        up_x = self.up(norm2_x)
        ffn_output = self.down(gate_x * up_x)
        out = ffn_output + norm2_x

        if padding_mask is not None:
            out = out * padding_mask
        return out


class HydraBlock(nn.Module):
    def __init__(self,
                 dim,
                 dt_rank,
                 dim_inner,
                 d_state,
                 args
                 ):
        super().__init__()
        self.multiheadMamba = MultiHeadMamba(dim, dt_rank, dim_inner, d_state, args)
        self.gate = nn.Linear(dim, dim, bias=False)  # W_gate
        self.up = nn.Linear(dim, dim, bias=False)  # W_up
        self.down = nn.Linear(dim, dim, bias=False)  # W_down
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=args.hydra_dropout_rate)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, padding_mask):
        norm_x = self.norm1(self.dropout(x))
        mamba_x = x + self.multiheadMamba(norm_x, x, padding_mask)

        # FFN with pre-norm
        norm2_x = self.norm2(mamba_x)
        gate_x = self.activation(self.gate(norm2_x))
        up_x = self.up(norm2_x)
        ffn_output = self.down(gate_x * up_x)
        out = ffn_output + mamba_x

        if padding_mask is not None:
            out = out * padding_mask
        return out


class MultiScaleUserEncoder(nn.Module):
    """
    Captures both short-term and long-term user interests through multi-scale modeling.
    This helps improve NDCG by better understanding user preference patterns at different time scales.
    """

    def __init__(self, dim, dt_rank, dim_inner, d_state, args):
        super().__init__()
        self.args = args
        self.hidden_dim = dim

        # Short-term encoder for recent 10-15 items
        self.short_term_encoder = HydraBlock(dim, dt_rank, dim_inner, d_state, args)

        # Mid-term encoder for recent 20-30 items
        self.mid_term_encoder = HydraBlock(dim, dt_rank, dim_inner, d_state, args)

        # Long-term encoder for full sequence
        self.long_term_encoder = HydraBlock(dim, dt_rank, dim_inner, d_state, args)

        # Interest fusion with cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=args.hydra_num_heads,
            dropout=args.hydra_dropout_rate,
            batch_first=True
        )

        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

        # Final projection
        self.output_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, sequence, padding_mask=None):
        batch_size, seq_len, hidden_dim = sequence.shape

        # Define time windows
        short_window = min(10, seq_len)
        mid_window = min(25, seq_len)

        # Extract different time scales (from the end of sequence)
        short_seq = sequence[:, -short_window:, :]
        mid_seq = sequence[:, -mid_window:, :]
        long_seq = sequence

        # Create corresponding masks
        if padding_mask is not None:
            short_mask = padding_mask[:, -short_window:, :]
            mid_mask = padding_mask[:, -mid_window:, :]
            long_mask = padding_mask
        else:
            short_mask = mid_mask = long_mask = None

        # Encode at different scales
        short_repr = self.short_term_encoder(short_seq, short_mask)
        mid_repr = self.mid_term_encoder(mid_seq, mid_mask)
        long_repr = self.long_term_encoder(long_seq, long_mask)

        # Get the last hidden state from each scale
        short_final = short_repr[:, -1:, :]  # [B, 1, D]
        mid_final = mid_repr[:, -1:, :]  # [B, 1, D]
        long_final = long_repr[:, -1:, :]  # [B, 1, D]

        # Cross-attention: short-term queries attend to long-term context
        attended_repr, _ = self.cross_attention(
            query=short_final,
            key=long_repr,
            value=long_repr
        )

        # Adaptive gating for fusion
        concat_repr = torch.cat([short_final, mid_final, long_final], dim=-1)  # [B, 1, D*3]
        gate = self.fusion_gate(concat_repr)  # [B, 1, D]

        # Weighted fusion
        fused = gate * attended_repr + (1 - gate) * long_final
        fused = self.norm(fused)
        fused = self.output_proj(fused)

        # Expand back to full sequence length for compatibility
        fused_expanded = fused.expand(-1, seq_len, -1)

        # Combine with original long-term representation
        return long_repr + 0.5 * fused_expanded




class Hydra(nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, shared_embeddings=None):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.args = args

        if shared_embeddings is not None:
            self.shared_embeddings = shared_embeddings
        else:
            from embedding import SharedEmbeddingModule, SharedEmbeddingConfig
            SharedEmbeddingConfig(args)  # Set up shared_hidden_units
            self.shared_embeddings = SharedEmbeddingModule(user_num, item_num, feat_statistics, feat_types, args)

            # Project shared embeddings to Hydra's expected dimension if different
        if args.shared_hidden_units != args.hydra_hidden_units:
            self.embedding_projection = nn.Linear(args.shared_hidden_units, args.hydra_hidden_units)
        else:
            self.embedding_projection = nn.Identity()

        self.hydra_projection = nn.Linear(args.shared_hidden_units * 2, args.hydra_hidden_units)
        if not hasattr(self.shared_embeddings, 'hydra_projection'):
            self.shared_embeddings.hydra_projection = self.hydra_projection

        self.time_span_emb = torch.nn.Embedding(
            getattr(args, 'time_span', 512),
            args.hydra_hidden_units,
            padding_idx=0,
        )

        # HSTU layers
        self.hydra = torch.nn.ModuleList()

        self.multi_scale_encoder = MultiScaleUserEncoder(
            dim=args.hydra_hidden_units,
            dt_rank=16,
            dim_inner=4 * args.hydra_hidden_units,
            d_state=16,
            args=args
        )

        self.hidden_units = args.hydra_hidden_units
        for _ in range(args.hydra_num_blocks):
            hydra = HydraBlock(
                dim=args.hydra_hidden_units,
                dt_rank=16,
                dim_inner=4 * args.hydra_hidden_units,
                d_state=16,
                args=args
            )
            self.hydra.append(hydra)

        self.hydra_first = HydraBlockFirst(
            dim=args.hydra_hidden_units,
            dt_rank=16,
            dim_inner=4 * args.hydra_hidden_units,
            d_state=16,
            args=args
        )

        self.position_aware_encoder = PositionAwareEncoder(args.hydra_hidden_units, max_len=args.maxlen + 1)
        self.enchanced_loss = EnhancedTrainingLoss(args)
    def log2feats(self, log_seqs, seq_feature, mask, seq_timestamp, action_type, dwell_bins):
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        padding_mask = (log_seqs != 0).unsqueeze(-1).float()

        if maxlen > self.maxlen + 1:
            log_seqs = log_seqs[:, -self.maxlen - 1:]
            seq_feature = [feat[-self.maxlen - 1:] for feat in seq_feature]
            maxlen = self.maxlen + 1

        # Step 1: Create the positional sequence tensor
        poss = (
            torch.arange(1, maxlen + 1, device=self.dev)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        poss *= (log_seqs != 0)

        seqs_concat = self.shared_embeddings.get_complete_embeddings(
            log_seqs, seq_feature, mask=mask, include_user=True,
            fusion_style='early', model_style='hydra', pos_seq=poss
        )

        # Step 3: CRITICAL FIX - Project the 120-dim tensor down to the model's hidden size (60-dim) FIRST.
        seqs = self.hydra_projection(seqs_concat)
        # Now, `seqs` has the correct shape: [Batch, SeqLen, 60]

        # Step 4: Now that `seqs` is 60-dim, add all other 60-dim embeddings.
        seqs *= self.args.hydra_hidden_units ** 0.5

        # Add time and action embeddings
        time_intervals = torch.zeros_like(seq_timestamp, device=self.dev)
        time_intervals[:, 1:] = seq_timestamp[:, 1:] - seq_timestamp[:, :-1]
        prev_is_padding = torch.zeros_like(log_seqs, dtype=torch.bool, device=self.dev)
        prev_is_padding[:, 1:] = log_seqs[:, :-1] == 0
        time_intervals[prev_is_padding] = 0

        time_span = getattr(self.args, 'time_span', 512)
        log_time_intervals = torch.log(time_intervals.float() + 1.0)
        time_bins = torch.clamp(log_time_intervals.long(), max=time_span - 1)
        time_bins = time_bins * (log_seqs != 0).long()

        dwell_emb = self.shared_embeddings.dwell_time_emb(dwell_bins)

        # All these additions will now work because all tensors are 60-dim
        seqs += self.embedding_projection(self.shared_embeddings.time_span_emb(time_bins))
        seqs += self.embedding_projection(self.shared_embeddings.action_emb(action_type))
        seqs += self.embedding_projection(dwell_emb)

        # Apply padding mask after all additions are complete
        seqs = seqs * padding_mask


        # Step 5: Pass the final 60-dim sequence to the Hydra blocks
        seqs = self.hydra_first(seqs, padding_mask=padding_mask)
        for hydra_block in self.hydra:
            seqs = hydra_block(x=seqs, padding_mask=padding_mask)

        seqs = self.multi_scale_encoder(seqs, padding_mask)

        # Final mask application
        seqs = seqs * padding_mask
        return seqs

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature,
                neg_feature, timestamp, action_type, dwell_bins):

        log_feats = self.log2feats(user_item, seq_feature, mask, timestamp, action_type, dwell_bins)
        item_table = self.shared_embeddings.item_emb.weight
        loss_mask = (next_mask == 1).to(self.dev)


        pos_embs = self.shared_embeddings.get_complete_embeddings(
            pos_seqs, pos_feature, mask, include_user=False, fusion_style='early', model_style='hydra'
        )
        neg_embs = self.shared_embeddings.get_complete_embeddings(
            neg_seqs, neg_feature, mask, include_user=False, fusion_style='early', model_style='hydra'
        )
        user_embeddings = log_feats[:, -1, :]  # Last position embeddings

        # Option 1: Similarity-based hard negatives
        miner = HardNegativeMiner(self, item_table, top_k=50)
        hard_neg_indices = miner.mine_hard_negatives_similarity(
            user_embeddings,
            positive_items=pos_seqs.squeeze(-1),  # Remove sequence dim if single item
            exclude_items=user_item  # Exclude items from user's history
        )

        # Expand to match sequence length for compatibility
        seq_len = log_feats.shape[1]
        hard_negatives = hard_neg_indices.unsqueeze(1).expand(-1, seq_len, -1)
        pos_embs = self.hydra_projection(pos_embs)
        neg_embs = self.hydra_projection(neg_embs)
        loss = self.enchanced_loss(
            user_embs=log_feats,
            pos_embs=pos_embs,
            neg_embs = neg_embs,
            hard_negatives=hard_negatives,
            loss_mask=loss_mask,
            item_table=item_table
        )
        return loss

    def predict(self, log_seqs, seq_feature, mask, seq_timestamp, action_type):
        """Predict next item using shared embeddings"""
        log_feats = self.log2feats(log_seqs, seq_feature, mask, seq_timestamp, action_type)
        final_feat = log_feats[:, -1, :]

        # Use shared item embeddings for prediction
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))

    def get_all_logits(self, seq, seq_feat, token_type, seq_timestamp, action_type):
        """
        HELPER FOR EVALUATION.
        Gets logits for ALL positions. To be used inside torch.no_grad().
        """
        log_feats = self.log2feats(seq, seq_feat, token_type, seq_timestamp, action_type)  # Shape: (B, L, D)
        logits = F.linear(log_feats, self.item_emb.weight)  # Shape: (B, L, V)
        return logits

    def rerank_candidates(self, user_item, seq_feature, mask, ts, action_type, candidate_ids):
        """Rerank candidates using shared item embeddings"""
        # Get user embedding
        log_feats = self.log2feats(user_item, seq_feature, mask, ts, action_type)
        user_embedding = log_feats[:, -1, :]

        # Get candidate embeddings using shared embeddings
        shared_item_emb = self.shared_embeddings.item_emb(candidate_ids)
        candidate_embs = self.embedding_projection(shared_item_emb)

        # Calculate scores
        rerank_scores = torch.bmm(
            user_embedding.unsqueeze(1),
            candidate_embs.transpose(1, 2)
        ).squeeze(1)

        return rerank_scores

    def get_features_for_ids(self, item_ids):
        """
        Placeholder for a fast feature lookup system.
        In a real system, this would query a key-value store like Redis or a memory-mapped file.
        For this code, it will look into the dataset's item_feat_dict.
        """
        # This assumes your Hydra model has access to the dataset's feature dictionary.
        # You might need to pass `dataset.item_feat_dict` to the Hydra model during initialization.
        batch_features = []
        for item_id in item_ids:
            if str(item_id) in self.item_feat_dict:
                feat = self.fill_missing_feat(self.item_feat_dict[str(item_id)], item_id)
            else:
                # Handle the case where a candidate item might not have features (e.g., it's new)
                feat = self.feature_default_value
            batch_features.append(feat)
        return batch_features

    def get_user_embedding(self, seq, seq_feat, token_type, seq_timestamp, action_type, dwell_bins):
        """
        HELPER FOR EVALUATION.
        Gets logits for ALL positions. To be used inside torch.no_grad().
        """
        log_feats = self.log2feats(seq, seq_feat, token_type, seq_timestamp, action_type, dwell_bins)  # Shape: (B, L, D)
        user_embedding = log_feats[:, -1, :]
        return user_embedding

def _sample_inbatch_indices_no_self(M, rows, T, device):
    """
    Sample indices in [0..M-1] excluding self for each row index in `rows`.
    Returns a tensor of shape (len(rows), T) of indices sampled WITH REPLACEMENT.
    (When M-1 >= T we sample T values; otherwise caller should not call this function.)
    rows: 1D LongTensor with global row indices (values in 0..M-1)
    """
    # r in [0, M-2], shape (len(rows), T)
    r = torch.randint(0, M - 1, (rows.size(0), T), device=device)
    # If r >= row_index we shift +1 to skip the self index
    # Broadcast rows to (len(rows), 1) then compare
    shifted = (r >= rows.unsqueeze(1)).long()
    indices = r + shifted  # now values in [0..M-1] except row
    return indices  # (len(rows), T)


def _all_inbatch_indices_excluding_self(M, device):
    """
    Return for each i in [0..M-1] the list of all indices excluding i.
    Result shape: (M, M-1)
    """
    base = torch.arange(M, device=device).unsqueeze(0).repeat(M, 1)   # (M, M)
    rows = torch.arange(M, device=device).unsqueeze(1)               # (M, 1)
    mask = base != rows                                              # (M, M)
    # base[mask] will flatten, but we can reshape into (M, M-1)
    return base[mask].view(M, M - 1)                                 # (M, M-1)


def compute_mixed_contrastive_loss(
        log_feats,  # [B, L, D]
        pos_embs,  # [B, L, D]
        loss_mask,  # [B, L]  0/1 mask for valid positions
        item_table=None,  # [num_items+1, D] (tensor on same device) or None
        hard_negatives=None,  # [B, L, K] hard negative item IDs, or None
        T=512,  # total number of negatives per query
        hard_ratio=0.3,  # fraction of negatives that should be hard negatives
        temperature=0.05,
        normalize=True,
        q_chunk=64,
        device=None
):
    """
    Compute InfoNCE loss mixing in-batch negatives with hard negatives.

    Args:
        hard_negatives: [B, L, K] tensor of hard negative item IDs for each position
        hard_ratio: fraction of T negatives that should be hard negatives (0.0-1.0)
    """
    if device is None:
        device = log_feats.device

    # Flatten valid positions
    mask = loss_mask.bool()
    q = log_feats[mask]  # [M, D]
    k = pos_embs[mask]  # [M, D]
    M = q.shape[0]
    if M == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if normalize:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

    # Calculate how many hard vs in-batch negatives we need
    num_hard = int(T * hard_ratio)
    num_inbatch = T - num_hard

    # Flatten hard negatives if provided
    hard_neg_ids = None
    if hard_negatives is not None:
        hard_neg_ids = hard_negatives[mask]  # [M, K]

    # Process chunks
    loss_list = []
    rows = torch.arange(M, device=device)

    for start in range(0, M, q_chunk):
        end = min(start + q_chunk, M)
        q_chunk_tensor = q[start:end]  # [qC, D]
        rows_chunk = rows[start:end]  # [qC]
        chunk_size = end - start

        neg_embs_list = []

        # 1. Add hard negatives if available
        if hard_neg_ids is not None and num_hard > 0:
            chunk_hard_ids = hard_neg_ids[start:end]  # [qC, K]
            # Sample num_hard negatives from available hard negatives
            K = chunk_hard_ids.shape[1]
            if K >= num_hard:
                # Randomly sample num_hard from K available
                sample_indices = torch.randint(0, K, (chunk_size, num_hard), device=device)
                sampled_hard_ids = torch.gather(chunk_hard_ids, 1, sample_indices)
            else:
                # Use all K hard negatives and repeat some if needed
                repeat_times = (num_hard + K - 1) // K  # Ceiling division
                repeated_hard_ids = chunk_hard_ids.repeat(1, repeat_times)
                sampled_hard_ids = repeated_hard_ids[:, :num_hard]

            # Get embeddings for hard negatives
            hard_neg_embs = item_table[sampled_hard_ids]  # [qC, num_hard, D]
            if normalize:
                hard_neg_embs = F.normalize(hard_neg_embs, p=2, dim=-1)
            neg_embs_list.append(hard_neg_embs)

        # 2. Add in-batch negatives
        if num_inbatch > 0:
            if M - 1 >= num_inbatch:
                # Sample in-batch negatives (excluding self)
                inbatch_indices = _sample_inbatch_indices_no_self(M, rows_chunk, num_inbatch, device)
                inbatch_neg_embs = k[inbatch_indices]  # [qC, num_inbatch, D]
            else:
                # Use all available in-batch negatives
                all_inbatch = _all_inbatch_indices_excluding_self(M, device)[start:end]
                inbatch_neg_embs = k[all_inbatch]  # [qC, M-1, D]

                # Pad to num_inbatch if needed
                if inbatch_neg_embs.shape[1] < num_inbatch:
                    padding_needed = num_inbatch - inbatch_neg_embs.shape[1]
                    if item_table is not None:
                        # Fill remaining with random global negatives
                        num_items = item_table.shape[0] - 1
                        random_ids = torch.randint(1, num_items + 1, (chunk_size, padding_needed), device=device)
                        random_embs = item_table[random_ids]
                        if normalize:
                            random_embs = F.normalize(random_embs, p=2, dim=-1)
                        inbatch_neg_embs = torch.cat([inbatch_neg_embs, random_embs], dim=1)
                    else:
                        # Repeat last negatives
                        pad = inbatch_neg_embs[:, -1:, :].expand(-1, padding_needed, -1)
                        inbatch_neg_embs = torch.cat([inbatch_neg_embs, pad], dim=1)

            neg_embs_list.append(inbatch_neg_embs)

        # 3. Combine all negatives
        if neg_embs_list:
            all_neg_embs = torch.cat(neg_embs_list, dim=1)  # [qC, T, D]
        else:
            # Fallback to random negatives
            num_items = item_table.shape[0] - 1
            random_ids = torch.randint(1, num_items + 1, (chunk_size, T), device=device)
            all_neg_embs = item_table[random_ids]
            if normalize:
                all_neg_embs = F.normalize(all_neg_embs, p=2, dim=-1)

        # 4. Build keys: positive + negatives
        pos_for_chunk = k[start:end].unsqueeze(1)  # [qC, 1, D]
        keys_chunk = torch.cat([pos_for_chunk, all_neg_embs], dim=1)  # [qC, T+1, D]

        # 5. Compute loss
        q_exp = q_chunk_tensor.unsqueeze(1)  # [qC, 1, D]
        logits_chunk = (q_exp * keys_chunk).sum(-1) / temperature  # [qC, T+1]
        logp = F.log_softmax(logits_chunk, dim=1)
        loss_chunk = -logp[:, 0]  # [qC]
        loss_list.append(loss_chunk)

    loss = torch.cat(loss_list, dim=0).mean()
    return loss


# Hard negative mining strategies
class HardNegativeMiner:
    def __init__(self, model, item_embeddings, top_k=100):
        self.model = model
        self.item_embeddings = item_embeddings  # Pre-computed item embeddings
        self.top_k = top_k

    def mine_hard_negatives_similarity(self, user_embeddings, positive_items, exclude_items=None):
        """
        Mine hard negatives based on similarity to user embeddings.
        Returns items most similar to user but not positive.
        """
        batch_size = user_embeddings.shape[0]
        device = user_embeddings.device

        # Compute similarities with all items
        similarities = torch.mm(user_embeddings, self.item_embeddings.t())  # [B, num_items]

        # Mask out positive items and excluded items
        mask = torch.ones_like(similarities, dtype=torch.bool)
        for b in range(batch_size):
            if positive_items is not None:
                mask[b, positive_items[b]] = False
            if exclude_items is not None:
                mask[b, exclude_items[b]] = False

        similarities[~mask] = float('-inf')

        # Get top-k most similar (hard negatives)
        _, hard_neg_indices = torch.topk(similarities, self.top_k, dim=1)
        return hard_neg_indices

    def mine_hard_negatives_popular(self, user_embeddings, item_popularity, exclude_items=None):
        """
        Mine hard negatives from popular items (popularity bias).
        """
        batch_size = user_embeddings.shape[0]
        device = user_embeddings.device

        # Sample from popular items
        popular_probs = F.softmax(item_popularity.float(), dim=0)
        hard_negatives = torch.multinomial(popular_probs, self.top_k * batch_size, replacement=True)
        hard_negatives = hard_negatives.view(batch_size, self.top_k)

        return hard_negatives


class AdaptiveHardNegativeSampler(nn.Module):
    """
    Adaptive hard negative sampling that adjusts difficulty based on training progress
    """

    def __init__(self, initial_ratio=0.3, max_ratio=0.7):
        super().__init__()
        self.initial_ratio = initial_ratio
        self.max_ratio = max_ratio
        self.register_buffer('training_step', torch.tensor(0.0))

    def get_current_ratio(self, max_steps=10000):
        # Gradually increase hard negative ratio during training
        progress = min(self.training_step / max_steps, 1.0)
        return self.initial_ratio + (self.max_ratio - self.initial_ratio) * progress

    def update_step(self):
        self.training_step += 1


class PositionAwareEncoder(nn.Module):
    """
    Enhanced position encoding that considers item position importance
    """

    def __init__(self, hidden_dim, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Learnable position importance weights
        self.position_weights = nn.Parameter(torch.ones(max_len))

        # Position-aware attention
        self.position_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.2,
            batch_first=True
        )

    def forward(self, sequence, positions):
        """
        Args:
            sequence: [B, L, D]
            positions: [B, L] - position indices
        """
        batch_size, seq_len, _ = sequence.shape

        # Apply position-based weights
        pos_weights = F.softmax(self.position_weights[:seq_len], dim=0)
        weighted_sequence = sequence * pos_weights.view(1, -1, 1)

        # Position-aware self-attention
        attended_seq, _ = self.position_attention(
            weighted_sequence, weighted_sequence, weighted_sequence
        )

        return sequence + attended_seq


class EnhancedTrainingLoss(nn.Module):
    """
    Multi-component loss function for better ranking performance
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ranking_loss_weight = getattr(args, 'ranking_loss_weight', 0.0)
        self.contrastive_loss_weight = getattr(args, 'contrastive_loss_weight', 1.0)
        self.diversity_loss_weight = getattr(args, 'diversity_loss_weight', 0.0)
        self.adaptive_hard_sampler = AdaptiveHardNegativeSampler()

    def compute_ranking_loss(self, user_embs, pos_embs, neg_embs, loss_mask):
        """Compute BPR-style ranking loss"""
        pos_scores = (user_embs * pos_embs).sum(dim=-1)
        neg_scores = (user_embs * neg_embs).sum(dim=-1)

        ranking_loss = -F.logsigmoid(pos_scores - neg_scores)
        return (ranking_loss * loss_mask).sum() / loss_mask.sum()

    def compute_diversity_loss(self, user_embs, item_embs):
        """Encourage diversity in user representations"""
        # Compute pairwise similarities
        user_sim = torch.mm(F.normalize(user_embs, p=2, dim=1),
                            F.normalize(user_embs, p=2, dim=1).t())

        # Penalize high similarities between different users
        mask = torch.eye(user_sim.size(0), device=user_sim.device)
        diversity_loss = (user_sim * (1 - mask)).abs().mean()

        return diversity_loss

    def forward(self, user_embs, pos_embs, neg_embs, hard_negatives,  loss_mask, item_table):
        total_loss = 0.0
        print(self.adaptive_hard_sampler.get_current_ratio())
        # Main contrastive loss (your existing implementation)
        contrastive_loss = compute_mixed_contrastive_loss(
            log_feats=user_embs,
            pos_embs=pos_embs,
            loss_mask=loss_mask,
            item_table=item_table,
            hard_negatives=hard_negatives,
            hard_ratio=0.3,
            # You can control all options from your args
            T=512,
            temperature=0.05,
            q_chunk=256,
            device=self.args.device
        )
        self.adaptive_hard_sampler.update_step()
        total_loss += self.contrastive_loss_weight * contrastive_loss

        # Additional ranking loss
        if neg_embs is not None:
            ranking_loss = self.compute_ranking_loss(
                user_embs[:, -1, :], pos_embs[:, -1, :], neg_embs[:, -1, :], loss_mask[:, -1]
            )
            total_loss += self.ranking_loss_weight * ranking_loss

        # Diversity loss
        diversity_loss = self.compute_diversity_loss(
            user_embs[:, -1, :], item_table
        )
        total_loss += self.diversity_loss_weight * diversity_loss

        return total_loss
