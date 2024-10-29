import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataset import note_event_order, note_num_tokens
from models.transformers import DecTransformer, top_p


class NoteTupleEmbedding(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(num_tokens[event], dim, padding_idx=0) for event in note_event_order])
        for emb in self.embs:
            nn.init.kaiming_normal_(emb.weight)

    def forward(self, x):
        return torch.sum(torch.stack([emb(x[:, :, i]) for i, emb in enumerate(self.embs)], dim=-1), dim=-1)


class NoteTupleHead(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.to_logits = nn.ModuleList([nn.Linear(dim, num_tokens[event]) for event in note_event_order])

    def forward(self, x):
        return tuple(head(x) for head in self.to_logits)


class NoteDecTransformer(DecTransformer):
    def __init__(self, cfg, pad_value=0):
        super().__init__(cfg=cfg, pad_value=pad_value)
        self.token_emb = NoteTupleEmbedding(note_num_tokens, cfg.dim)
        self.to_logits = NoteTupleHead(cfg.dim, note_num_tokens)


class WrapperNoteDecTransformer(nn.Module):
    def __init__(self, cfg, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.rel_time = cfg.rel_time
        self.rel_pitch = cfg.rel_pitch
        self.decompose = cfg.decompose
        self.net = NoteDecTransformer(cfg=cfg, pad_value=pad_value)

    def generate(self, prompts, seq_len=1024, eos_token=3, sampling_method="top_p"):
        out = prompts
        cache = None
        if sampling_method == "top_p":
            sampling_func = top_p
        else:
            raise ValueError("Invalid sampling method")

        seq_len = seq_len - out.shape[1]
        out = torch.where(out == self.ignore_index, self.pad_value, out)

        for _ in range(seq_len):
            if self.rel_time or self.rel_pitch:
                use_cache = cache is not None
                rel_structure_matrix = self.make_relative_matrix(out, rel_time=self.rel_time, rel_pitch=self.rel_pitch, use_cache=use_cache, decompose=self.decompose)
            else:
                rel_structure_matrix = None

            logits, new_cache = self.net(out, rel_structure_matrix=rel_structure_matrix, return_intermediates=True, cache=cache)
            cache = new_cache
            for i in range(6):
                filterd_logits = sampling_func(logits[i][:, -1])
                probs = F.softmax(filterd_logits, dim=-1)
                if i == 0:
                    sample = torch.multinomial(probs, 1)
                else:
                    sample_attr = torch.multinomial(probs, 1)
                    sample = torch.cat((sample, sample_attr), dim=-1)

            # out.shape = (b, n, 6), sample.shape = (b, 6)
            out = torch.cat((out, sample.unsqueeze(1)), dim=1)
            is_eos_tokens = out[:, :, 0] == eos_token

            if is_eos_tokens.any(dim=-1).all():
                break

        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
        mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
        out = out.masked_fill(mask.unsqueeze(-1), self.pad_value)

        return out

    def forward(self, x):
        ignore_index = self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.rel_time or self.rel_pitch:
            rel_structure_matrix = self.make_relative_matrix(inp, rel_time=self.rel_time, rel_pitch=self.rel_pitch, decompose=self.decompose)
        else:
            rel_structure_matrix = None

        logits, _ = self.net(inp, rel_structure_matrix=rel_structure_matrix)

        losses = [F.cross_entropy(rearrange(logits[i], "b n c -> b c n"), target[:, :, i], ignore_index=ignore_index) for i in range(6)]

        losses = torch.stack(losses).unsqueeze(0)
        return losses

    def make_relative_matrix(self, inp, rel_time=False, rel_pitch=False, use_cache=False, decompose=True):
        """
        This function generates time and(or) pitch relative matrix.
        When use_cache is False, the shape of the matrix is (b, n, n), where b is batch size, n is the length of the sequence.
        When use_cache is True, the shape of the matrix is (b, 1, n) which is the relative of the last token.
        """
        pad_value = self.pad_value
        b, n, _ = inp.shape
        rel_structure_matrix = {}

        pad_pos = inp[:, :, 1] == pad_value
        if use_cache:
            mask = pad_pos.view(b, 1, n)
        else:
            mask = torch.triu(torch.ones(n, n, device=inp.device), diagonal=1).bool().repeat(b, 1, 1)
            mask[pad_pos] = True
            mask[pad_pos.view(b, 1, n).repeat(1, n, 1)] = True

        if rel_time:
            if decompose:
                if use_cache:
                    rsm_bar = inp[:, :, 1].view(b, 1, n) - inp[:, -1, 1].view(b, 1, 1) + 15
                    rsm_pos = inp[:, :, 2].view(b, 1, n) - inp[:, -1, 2].view(b, 1, 1) + 47
                else:
                    rsm_bar = inp[:, :, 1].view(b, 1, n) - inp[:, :, 1].view(b, n, 1) + 15
                    rsm_pos = inp[:, :, 2].view(b, 1, n) - inp[:, :, 2].view(b, n, 1) + 47
                rsm_bar[mask] = -1
                rsm_bar[~mask] = torch.clamp(rsm_bar[~mask], min=0, max=15)
                rsm_pos[mask] = -1
                rsm_pos[~mask] = torch.clamp(rsm_pos[~mask], min=0, max=94)
                rel_structure_matrix["bar"] = rsm_bar
                rel_structure_matrix["position"] = rsm_pos
            else:
                time = (inp[:, :, 1] - 1) * 48 + inp[:, :, 2]
                if use_cache:
                    rsm_time = time.view(b, 1, n) - time[:, -1].view(b, 1, 1) + 767
                else:
                    rsm_time = time.view(b, 1, n) - time.view(b, n, 1) + 767
                rsm_time[mask] = -1
                rsm_time[~mask] = torch.clamp(rsm_time[~mask], min=0, max=767)
                rel_structure_matrix["time"] = rsm_time

        if rel_pitch:
            if decompose:
                if use_cache:
                    rsm_pitch = inp[:, :, 4].view(b, 1, n) - inp[:, -1, 4].view(b, 1, 1)
                else:
                    rsm_pitch = inp[:, :, 4].view(b, 1, n) - inp[:, :, 4].view(b, n, 1)
                rsm_octave = rsm_pitch // 12 + 11
                rsm_semi = rsm_pitch % 12
                rsm_octave[mask] = -1
                rsm_octave[~mask] = torch.clamp(rsm_octave[~mask], min=0, max=21)
                rsm_semi[mask] = -1
                rel_structure_matrix["octave"] = rsm_octave
                rel_structure_matrix["semitone"] = rsm_semi
            else:
                pitch = inp[:, :, 4]
                if use_cache:
                    rsm_pitch = pitch.view(b, 1, n) - pitch[:, -1].view(b, 1, 1) + 127
                else:
                    rsm_pitch = pitch.view(b, 1, n) - pitch.view(b, n, 1) + 127
                rsm_pitch[mask] = -1
                rsm_pitch[~mask] = torch.clamp(rsm_pitch[~mask], min=0, max=254)
                rel_structure_matrix["pitch"] = rsm_pitch

        return rel_structure_matrix
