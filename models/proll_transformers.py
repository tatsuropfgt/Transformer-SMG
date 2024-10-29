import torch
from torch import nn
from torch.nn import functional as F

from dataset import proll_num_token, proll_seq_len
from models.transformers import DecTransformer


class PianoRollDecTransformer(DecTransformer):
    def __init__(self, cfg, pad_value=0):
        super().__init__(cfg=cfg, pad_value=pad_value, max_seq_len=proll_seq_len)
        self.token_emb = nn.Linear(proll_num_token, cfg.dim)
        self.to_logits = nn.Linear(cfg.dim, proll_num_token)


class WrapperPianoRollDecTransformer(nn.Module):
    def __init__(self, cfg, pad_value=0):
        super().__init__()
        self.decompose = cfg.decompose
        if cfg.rel_time:
            self.init_time_relative_matrix(proll_seq_len)
        self.rel_time = cfg.rel_time
        self.rel_pitch = cfg.rel_pitch
        self.net = PianoRollDecTransformer(cfg=cfg, pad_value=pad_value)

    def generate(self, prompts, seq_len=769, threshold=0.5):
        # batch x seq_len x (128 x 3 + 1)
        out = prompts
        cache = None

        seq_len = seq_len - out.shape[1]

        for _ in range(seq_len):
            rel_structure_matrix = {}
            if self.rel_time or self.rel_pitch:
                use_cache = cache is not None
                rel_structure_matrix = self.make_relative_matrix(out, rel_time=self.rel_time, rel_pitch=self.rel_pitch, use_cache=use_cache)
            else:
                rel_structure_matrix = None

            logits, new_cache = self.net(out, rel_structure_matrix=rel_structure_matrix, return_intermediates=True, cache=cache)
            cache = new_cache
            logits = logits[:, -1]  # batch x (128 x 3 + 1)
            probs = F.sigmoid(logits)
            # if the probability is more than threshold then it is 1, otherwise 0
            sample = (probs > threshold).float()
            out = torch.cat((out, sample.unsqueeze(1)), dim=1)

        return out

    def forward(self, x, return_outputs=False):
        inp, target = x[:, :-1], x[:, 1:]

        if self.rel_time or self.rel_pitch:
            rel_structure_matrix = self.make_relative_matrix(inp, rel_time=self.rel_time, rel_pitch=self.rel_pitch)
        else:
            rel_structure_matrix = None

        logits, cache = self.net(inp, rel_structure_matrix=rel_structure_matrix, return_intermediates=return_outputs)

        loss = F.binary_cross_entropy_with_logits(logits, target.float())

        if return_outputs:
            return loss, (logits, cache)
        return loss.unsqueeze(0)

    def init_time_relative_matrix(self, n):
        if self.decompose:
            bar_idx = torch.zeros(n)
            position_idx = torch.zeros(n)
            for i in range(n - 1):
                bar_idx[i + 1] = i // 48
                position_idx[i + 1] = i % 48
            rsm_bar = bar_idx.view(1, n) - bar_idx.view(n, 1) + 15
            rsm_pos = position_idx.view(1, n) - position_idx.view(n, 1) + 47
            mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
            mask[0] = True
            mask[:, 0] = True
            rsm_bar[mask] = -1
            rsm_pos[mask] = -1
            self.register_buffer("rsm_bar", rsm_bar.to(torch.int32), persistent=False)
            self.register_buffer("rsm_pos", rsm_pos.to(torch.int32), persistent=False)
        else:
            time_idx = torch.zeros(n)
            rsm_time = time_idx.view(1, n) - time_idx.view(n, 1) + 767
            mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
            mask[0] = True
            mask[:, 0] = True
            rsm_time[mask] = -1
            self.register_buffer("rsm_time", rsm_time.to(torch.int32), persistent=False)

    def make_relative_matrix(self, inp, rel_time=False, rel_pitch=False, use_cache=False):
        b, n, _ = inp.shape
        rel_structure_matrix = {}
        device = inp.device
        decompose = self.decompose

        if rel_time:
            if decompose:
                if use_cache:
                    rsm_bar = self.rsm_bar[n - 1, :n].unsqueeze(0).repeat(b, 1, 1)
                    rsm_position = self.rsm_pos[n - 1, :n].unsqueeze(0).repeat(b, 1, 1)
                else:
                    rsm_bar = self.rsm_bar[:n, :n].unsqueeze(0).repeat(b, 1, 1)
                    rsm_position = self.rsm_pos[:n, :n].unsqueeze(0).repeat(b, 1, 1)
                rel_structure_matrix["bar"] = rsm_bar
                rel_structure_matrix["position"] = rsm_position
            else:
                if use_cache:
                    rsm_time = self.rsm_time[n - 1, :n].unsqueeze(0).repeat(b, 1, 1)
                else:
                    rsm_time = self.rsm_time[:n, :n].unsqueeze(0).repeat(b, 1, 1)
                rel_structure_matrix["time"] = rsm_time

        if rel_pitch:
            if decompose:
                melody = inp[:, :, 1:129]
                highest_indices = torch.where(melody == 1, torch.arange(128, device=device), torch.tensor(-100, device=device))
                highest_melody = torch.max(highest_indices, dim=2).values
                ignore_pos = highest_melody == -100
                if use_cache:
                    mask = ignore_pos.view(b, 1, n)
                    mask[highest_melody[:, -1] == -100] = True
                    rsm_pitch = highest_melody.view(b, 1, n) - highest_melody[:, -1].view(b, 1, 1)
                else:
                    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool().unsqueeze(0).repeat(b, 1, 1)
                    mask[ignore_pos] = True
                    mask[ignore_pos.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_pitch = highest_melody.view(b, 1, n) - highest_melody.view(b, n, 1)
                rsm_octave = rsm_pitch // 12 + 12
                rsm_semi = rsm_pitch % 12
                rsm_octave[mask] = -1
                rsm_semi[mask] = -1
                rel_structure_matrix["octave"] = rsm_octave
                rel_structure_matrix["semitone"] = rsm_semi
            else:
                melody = inp[:, :, 1:129]
                highest_indices = torch.where(melody == 1, torch.arange(128, device=device), torch.tensor(-100, device=device))
                highest_melody = torch.max(highest_indices, dim=2).values
                ignore_pos = highest_melody == -100
                if use_cache:
                    mask = ignore_pos.view(b, 1, n)
                    mask[highest_melody[:, -1] == -100] = True
                    rsm_pitch = highest_melody.view(b, 1, n) - highest_melody[:, -1].view(b, 1, 1) + 127
                else:
                    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool().unsqueeze(0).repeat(b, 1, 1)
                    mask[ignore_pos] = True
                    mask[ignore_pos.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_pitch = highest_melody.view(b, 1, n) - highest_melody.view(b, n, 1) + 127
                rsm_pitch[mask] = -1
                rel_structure_matrix["pitch"] = rsm_pitch

        return rel_structure_matrix
