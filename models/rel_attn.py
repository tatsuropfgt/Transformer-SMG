import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from models.hook import ForHook

BIAS_BAR = 7920  # 2^3 x 5 x 193
BIAS_POSITION = 8993  # 17 x 23 x 23
BIAS_OCTAVE = 9919  # 7 x 13 x 109
BIAS_SEMITONE = 10469  # 19 x 19 x 29
BIAS_TIME = 7920
BIAS_PITCH = 9919


class RelativeStructureAttention(nn.Module):
    def __init__(self, cfg, max_seq_len):
        super().__init__()
        self.heads = cfg.heads
        self.max_seq_len = max_seq_len
        self.rel_idx = cfg.rel_idx
        self.rel_time = cfg.rel_time
        self.rel_pitch = cfg.rel_pitch
        self.decompose = cfg.decompose
        self.hadamard = cfg.hadamard
        self.alpha = cfg.alpha if "alpha" in cfg else 1.0
        self.linear = cfg.linear if "linear" in cfg else False

        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        d_h = cfg.dim // cfg.heads
        if cfg.rel_idx:
            self.Er = nn.Parameter(torch.randn(max_seq_len, d_h))
        self.register_buffer("zeros_for_ignore", torch.zeros((1, d_h)))
        if cfg.rel_tp_sinusoidal:
            idx = torch.arange(d_h, device=self.Er.device)
        if cfg.rel_time:
            if cfg.decompose:
                if cfg.rel_tp_sinusoidal:
                    bar_idx = torch.arange(15, -1, -1, device=self.Er.device)
                    position_idx = torch.arange(47, -48, -1, device=self.Er.device)
                    angles_bar = 1 / torch.pow(BIAS_BAR, (2 * idx) / d_h)
                    angles_position = 1 / torch.pow(BIAS_POSITION, (2 * idx) / d_h)
                    wf_bar = torch.einsum("s, d -> s d", bar_idx, angles_bar)
                    wf_position = torch.einsum("s, d -> s d", position_idx, angles_position)
                    wf_bar[:, 0::2] = torch.sin(wf_bar[:, 0::2])
                    wf_bar[:, 1::2] = torch.cos(wf_bar[:, 1::2])
                    wf_position[:, 0::2] = torch.sin(wf_position[:, 0::2])
                    wf_position[:, 1::2] = torch.cos(wf_position[:, 1::2])
                    Er_bar = torch.cat([wf_bar, torch.zeros((1, d_h))])
                    Er_position = torch.cat([wf_position, torch.zeros((1, d_h))])
                    self.register_buffer("Er_bar", Er_bar)
                    self.register_buffer("Er_position", Er_position)
                else:
                    self.Er_bar = nn.Parameter(torch.randn(16, d_h))  # -15 ~ 0
                    self.Er_position = nn.Parameter(torch.randn(95, d_h))  # -47 ~ 47
                if cfg.linear:
                    self.rel_bar_linear = nn.Linear(d_h, d_h, bias=False)
                    self.rel_position_linear = nn.Linear(d_h, d_h, bias=False)
            else:
                if cfg.rel_tp_sinusoidal:
                    time_idx = torch.arange(767, -1, -1, device=self.Er.device)
                    angles_time = 1 / torch.pow(BIAS_TIME, (2 * idx) / d_h)
                    wf_time = torch.einsum("s, d -> s d", time_idx, angles_time)
                    wf_time[:, 0::2] = torch.sin(wf_time[:, 0::2])
                    wf_time[:, 1::2] = torch.cos(wf_time[:, 1::2])
                    Er_time = torch.cat([wf_time, torch.zeros((1, d_h))])
                    self.register_buffer("Er_time", Er_time)
                else:
                    self.Er_time = nn.Parameter(torch.randn(768, d_h))
                if cfg.linear:
                    self.rel_time_linear = nn.Linear(d_h, d_h, bias=False)
        if cfg.rel_pitch:
            if cfg.decompose:
                if cfg.rel_tp_sinusoidal:
                    octave_idx = torch.arange(11, -11, -1, device=self.Er.device)
                    semitone_idx = torch.arange(12, device=self.Er.device)
                    angles_octave = 1 / torch.pow(BIAS_OCTAVE, (2 * idx) / d_h)
                    angles_semitone = 1 / torch.pow(BIAS_SEMITONE, (2 * idx) / d_h)
                    wf_octave = torch.einsum("s, d -> s d", octave_idx, angles_octave)
                    wf_semitone = torch.einsum("s, d -> s d", semitone_idx, angles_semitone)
                    wf_octave[:, 0::2] = torch.sin(wf_octave[:, 0::2])
                    wf_octave[:, 1::2] = torch.cos(wf_octave[:, 1::2])
                    wf_semitone[:, 0::2] = torch.sin(wf_semitone[:, 0::2])
                    wf_semitone[:, 1::2] = torch.cos(wf_semitone[:, 1::2])
                    Er_octave = torch.cat([wf_octave, torch.zeros((1, d_h))])
                    Er_semitone = torch.cat([wf_semitone, torch.zeros((1, d_h))])
                    self.register_buffer("Er_octave", Er_octave)
                    self.register_buffer("Er_semitone", Er_semitone)
                else:
                    self.Er_octave = nn.Parameter(
                        torch.randn(24, d_h)
                    )  # -12 ~ 11 (C0 ~ G10)
                    self.Er_semitone = nn.Parameter(torch.randn(12, d_h))  # 0 ~ 11 (C ~ B)
                if cfg.linear:
                    self.rel_octave_linear = nn.Linear(d_h, d_h, bias=False)
                    self.rel_semitone_linear = nn.Linear(d_h, d_h, bias=False)
            else:
                if cfg.rel_tp_sinusoidal:
                    pitch_idx = torch.arange(127, -128, -1, device=self.Er.device)
                    angles_pitch = 1 / torch.pow(BIAS_PITCH, (2 * idx) / d_h)
                    wf_pitch = torch.einsum("s, d -> s d", pitch_idx, angles_pitch)
                    wf_pitch[:, 0::2] = torch.sin(wf_pitch[:, 0::2])
                    wf_pitch[:, 1::2] = torch.cos(wf_pitch[:, 1::2])
                    Er_pitch = torch.cat([wf_pitch, torch.zeros((1, d_h))])
                    self.register_buffer("Er_pitch", Er_pitch)
                else:
                    self.Er_pitch = nn.Parameter(torch.randn(255, d_h))
                if cfg.linear:
                    self.rel_pitch_linear = nn.Linear(d_h, d_h, bias=False)

        self.to_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.attn_hook = ForHook()
        self.normattn_hook = ForHook()

    def forward(self, x, rel_structure_matrix, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, query length (=1 when generation), key length)
        d - head dimension (dim // heads)
        rel_structure_matrix: dict(str, tensor.shape=(b, n, n))
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        QK_t = einsum("b h i d, b h j d -> b h i j", q, k)
        b, i, j = QK_t.shape[0], QK_t.shape[2], QK_t.shape[3]

        Srel = 0
        if self.rel_idx:
            start = self.max_seq_len - k.size(-2)
            Er = self.Er[start:, :]
            QEr = einsum("b h i d, j d -> b h i j", q, Er)
            if i != 1:
                Srel = self.skew(QEr)
            else:
                Srel = QEr

        device = q.device

        idx_0 = torch.arange(b, device=device).view(b, 1, 1, 1).expand(b, h, i, j)
        idx_1 = torch.arange(h, device=device).view(1, h, 1, 1).expand(b, h, i, j)
        idx_2 = torch.arange(i, device=device).view(1, 1, i, 1).expand(b, h, i, j)

        S_time = 0
        S_pitch = 0

        if self.rel_time:
            if self.decompose:
                Er_bar_0 = torch.cat([self.Er_bar, self.zeros_for_ignore])
                Er_position_0 = torch.cat([self.Er_position, self.zeros_for_ignore])
                if self.linear:
                    Er_bar_0 = self.rel_bar_linear(Er_bar_0)
                    Er_position_0 = self.rel_position_linear(Er_position_0)

                if self.hadamard:
                    Er_bar_position = einsum(
                        "s d, t d -> s t d", Er_bar_0, Er_position_0
                    )
                    QEr_bar_position = einsum(
                        "b h i d, s t d -> b h i s t", q, Er_bar_position
                    )
                    idx_3_bar = (
                        rel_structure_matrix["bar"].unsqueeze(1).expand(b, h, i, j)
                    )
                    idx_3_position = (
                        rel_structure_matrix["position"].unsqueeze(1).expand(b, h, i, j)
                    )
                    S_time = QEr_bar_position[
                        idx_0, idx_1, idx_2, idx_3_bar, idx_3_position
                    ]
                else:
                    QEr_bar = einsum("b h i d, s d -> b h i s", q, Er_bar_0)
                    idx_3 = rel_structure_matrix["bar"].unsqueeze(1).expand(b, h, i, j)
                    S_bar = QEr_bar[idx_0, idx_1, idx_2, idx_3]

                    QEr_position = einsum("b h i d, s d -> b h i s", q, Er_position_0)
                    idx_3 = (
                        rel_structure_matrix["position"].unsqueeze(1).expand(b, h, i, j)
                    )
                    S_position = QEr_position[idx_0, idx_1, idx_2, idx_3]
                    S_time = S_bar + S_position
            else:
                Er_time_0 = torch.cat([self.Er_time, self.zeros_for_ignore])
                if self.linear:
                    Er_time_0 = self.rel_time_linear(Er_time_0)
                QEr_time = einsum("b h i d, s d -> b h i s", q, Er_time_0)
                idx_3 = rel_structure_matrix["time"].unsqueeze(1).expand(b, h, i, j)
                S_time = QEr_time[idx_0, idx_1, idx_2, idx_3]

        if self.rel_pitch:
            if self.decompose:
                Er_octave_0 = torch.cat([self.Er_octave, self.zeros_for_ignore])
                Er_semitone_0 = torch.cat([self.Er_semitone, self.zeros_for_ignore])
                if self.linear:
                    Er_octave_0 = self.rel_octave_linear(Er_octave_0)
                    Er_semitone_0 = self.rel_semitone_linear(Er_semitone_0)
                if self.hadamard:
                    Er_octave_semitone = einsum(
                        "s d, t d -> s t d", Er_octave_0, Er_semitone_0
                    )
                    QEr_octave_semitone = einsum(
                        "b h i d, s t d -> b h i s t", q, Er_octave_semitone
                    )
                    idx_3_octave = (
                        rel_structure_matrix["octave"].unsqueeze(1).expand(b, h, i, j)
                    )
                    idx_3_semitone = (
                        rel_structure_matrix["semitone"].unsqueeze(1).expand(b, h, i, j)
                    )
                    S_pitch = QEr_octave_semitone[
                        idx_0, idx_1, idx_2, idx_3_octave, idx_3_semitone
                    ]
                else:
                    QEr_octave = einsum("b h i d, s d -> b h i s", q, Er_octave_0)
                    idx_3 = (
                        rel_structure_matrix["octave"].unsqueeze(1).expand(b, h, i, j)
                    )
                    S_octave = QEr_octave[idx_0, idx_1, idx_2, idx_3]

                    QEr_semitone = einsum("b h i d, s d -> b h i s", q, Er_semitone_0)
                    idx_3 = (
                        rel_structure_matrix["semitone"].unsqueeze(1).expand(b, h, i, j)
                    )
                    S_semitone = QEr_semitone[idx_0, idx_1, idx_2, idx_3]
                    S_pitch = S_octave + S_semitone
            else:
                Er_pitch_0 = torch.cat([self.Er_pitch, self.zeros_for_ignore])
                if self.linear:
                    Er_pitch_0 = self.rel_pitch_linear(Er_pitch_0)
                QEr_pitch = einsum("b h i d, s d -> b h i s", q, Er_pitch_0)
                idx_3 = rel_structure_matrix["pitch"].unsqueeze(1).expand(b, h, i, j)
                S_pitch = QEr_pitch[idx_0, idx_1, idx_2, idx_3]

        dots = ((QK_t) + self.alpha * (Srel + S_time + S_pitch)) / ((q.size(-1)) ** 0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(
                dots.shape[-2], dots.shape[-1], device=q.device
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        _ = self.attn_hook(attn=attn)
        _ = self.normattn_hook(attn=attn, value=v)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out, None

        return out, cached_kv

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class RelativeAttention(nn.Module):
    def __init__(self, cfg, max_seq_len):
        super().__init__()
        self.heads = cfg.heads
        self.max_seq_len = max_seq_len
        self.alpha = cfg.alpha if "alpha" in cfg else 1.0

        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.Er = nn.Parameter(torch.randn(max_seq_len, cfg.dim // cfg.heads))
        self.to_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.attn_hook = ForHook()
        self.normattn_hook = ForHook()

    def forward(self, x, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, query length (=1 when generation), key length)
        d - head dimension (dim // heads)
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        QK_t = einsum("b h i d, b h j d -> b h i j", q, k)

        start = self.max_seq_len - k.size(-2)
        Er_t = self.Er[start:, :]
        QEr = einsum("b h i d, j d -> b h i j", q, Er_t)

        if q.shape[-2] != 1:
            Srel = self.skew(QEr)
        else:
            Srel = QEr

        dots = (QK_t + self.alpha * Srel) / ((q.size(-1)) ** 0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(
                dots.shape[-2], dots.shape[-1], device=q.device
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        _ = self.attn_hook(attn=attn)
        _ = self.normattn_hook(attn=attn, value=v)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out, None

        return out, cached_kv

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class SimpleRelativeAttention(nn.Module):
    def __init__(self, cfg, max_seq_len):
        super().__init__()
        self.heads = cfg.heads
        self.max_seq_len = max_seq_len
        self.alpha = cfg.alpha if "alpha" in cfg else 1.0

        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.relative_emb = nn.Embedding(max_seq_len, cfg.heads)
        self.to_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, query length (=1 when generation), key length)
        d - head dimension (dim // heads)
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        QK_t = einsum("b h i d, b h j d -> b h i j", q, k)

        i_position = torch.arange(q.size(-2), device=q.device)[:, None]
        j_position = torch.arange(k.size(-2), device=q.device)[None, :]
        rel_pos = i_position - j_position
        rel_pos = rel_pos.clamp(0, self.max_seq_len - 1)

        Srel = self.relative_emb(rel_pos).permute(2, 0, 1).unsqueeze(0)

        dots = (QK_t + self.alpha * Srel) / ((q.size(-1)) ** 0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(
                dots.shape[-2], dots.shape[-1], device=q.device
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out, None

        return out, cached_kv

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


class SimpleRelativeStructureAttention(nn.Module):
    def __init__(self, cfg, max_seq_len):
        super().__init__()
        self.heads = cfg.heads
        self.max_seq_len = max_seq_len
        self.rel_idx = cfg.rel_idx
        self.rel_time = cfg.rel_time
        self.rel_pitch = cfg.rel_pitch
        self.hadamard = cfg.hadamard
        self.alpha = cfg.alpha if "alpha" in cfg else 1.0
        self.linear = cfg.linear if "linear" in cfg else False

        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.rel_idx:
            self.rel_idx_emb = nn.Embedding(max_seq_len, self.heads)
        if cfg.rel_time:
            self.rel_bar_emb = nn.Embedding(17, self.heads, padding_idx=0)
            self.rel_position_emb = nn.Embedding(96, self.heads, padding_idx=0)
        if cfg.rel_pitch:
            self.rel_octave_emb = nn.Embedding(25, self.heads, padding_idx=0)
            self.rel_semitone_emb = nn.Embedding(13, self.heads, padding_idx=0)

        self.to_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, rel_structure_matrix, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, query length (=1 when generation), key length)
        d - head dimension (dim // heads)
        rel_structure_matrix: dict(str, tensor.shape=(b, n, n))
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        QK_t = einsum("b h i d, b h j d -> b h i j", q, k)
        i, j = QK_t.shape[2], QK_t.shape[3]

        device = q.device

        i_position = torch.arange(i, device=device)[:, None]
        j_position = torch.arange(j, device=device)[None, :]
        rel_pos = i_position - j_position
        rel_pos = rel_pos.clamp(0, self.max_seq_len - 1)

        Srel = self.rel_idx_emb(rel_pos).permute(2, 0, 1).unsqueeze(0)

        S_time = 0
        S_pitch = 0

        if self.rel_time:
            S_bar = self.rel_bar_emb(rel_structure_matrix["bar"] + 1).permute(
                0, 3, 1, 2
            )
            S_position = self.rel_position_emb(
                rel_structure_matrix["position"] + 1
            ).permute(0, 3, 1, 2)

            S_time = S_bar + S_position  # b, h, i, j

        if self.rel_pitch:
            S_octave = self.rel_octave_emb(rel_structure_matrix["octave"] + 1).permute(
                0, 3, 1, 2
            )
            S_semitone = self.rel_semitone_emb(
                rel_structure_matrix["semitone"] + 1
            ).permute(0, 3, 1, 2)

            S_pitch = S_octave + S_semitone

        dots = ((QK_t) + self.alpha * (Srel + S_time + S_pitch)) / ((q.size(-1)) ** 0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(
                dots.shape[-2], dots.shape[-1], device=q.device
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out, None

        return out, cached_kv

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
