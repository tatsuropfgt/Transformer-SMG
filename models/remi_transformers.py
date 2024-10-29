import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import MAX_BAR, MAX_PITCH, MAX_POSITION, remi_indexer, remi_num_tokens
from models.transformers import DecTransformer, top_p


class WrapperREMIDecTransformer(nn.Module):
    def __init__(self, cfg, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.rel_time = cfg.rel_time
        self.rel_pitch = cfg.rel_pitch
        self.decompose = cfg.decompose
        self.net = DecTransformer(
            cfg=cfg,
            pad_value=pad_value,
            num_tokens=remi_num_tokens,
            max_seq_len=cfg.max_note_len * 4,
        )

    def generate(
        self,
        prompts,
        seq_len=4048,
        eos_token=remi_indexer["end-of-song"],
        sampling_method="top_p",
    ):
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
                rel_structure_matrix = self.make_relative_matrix(
                    out,
                    rel_time=self.rel_time,
                    rel_pitch=self.rel_pitch,
                    use_cache=use_cache,
                )
            else:
                rel_structure_matrix = None

            logits, new_cache = self.net(
                out,
                rel_structure_matrix=rel_structure_matrix,
                return_intermediates=True,
                cache=cache,
            )
            cache = new_cache

            filtered_logits = sampling_func(logits[:, -1])
            probs = F.softmax(filtered_logits, dim=-1)
            sample = torch.multinomial(probs, 1)

            # out.shape = (b, n), sample.shape = (b, 1)
            out = torch.cat((out, sample), dim=1)
            is_eos_tokens = out == eos_token

            if is_eos_tokens.any(dim=-1).all():
                break

        return out

    def forward(self, x):
        ignore_index = self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.rel_time or self.rel_pitch:
            rel_structure_matrix = self.make_relative_matrix(
                inp, rel_time=self.rel_time, rel_pitch=self.rel_pitch
            )
        else:
            rel_structure_matrix = None

        logits, _ = self.net(inp, rel_structure_matrix=rel_structure_matrix)

        loss = F.cross_entropy(
            logits.permute(0, 2, 1), target, ignore_index=ignore_index
        )

        return loss

    def make_relative_matrix(
        self, inp, rel_time=False, rel_pitch=False, use_cache=False):
        """
        This function generates time and(or) pitch relative matrix.
        When use_cache is False, the shape of the matrix is (b, n, n), where b is batch size, n is the length of the sequence.
        When use_cache is True, the shape of the matrix is (b, 1, n) which is the relative of the last token.
        """
        pad_value = self.pad_value
        b, n = inp.shape
        rel_structure_matrix = {}
        pad_pos = inp == pad_value
        device = inp.device
        decompose = self.decompose

        if rel_time:
            if decompose:
                # bar_0 の token_id
                bar_min_id = remi_indexer["bar_0"]
                bar_max_id = bar_min_id + MAX_BAR - 1
                # bar_X を持つトークンの位置
                bar_mask = (inp >= bar_min_id) & (inp <= bar_max_id)
                # bar_X を持つトークン位置にそのトークンの index を入れる
                bar_tokens_arange = torch.where(bar_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                # cummax により、すべてのトークン位置の bar_X への index を取得
                bar_tokens_idx = torch.cummax(bar_tokens_arange, dim=1)[0]
                bar = torch.where(
                    bar_tokens_idx == 0, torch.tensor(0), inp.gather(1, bar_tokens_idx)
                )
                # 0~15 に変換
                pre_bar = bar == 0
                mask_pos = pre_bar | pad_pos
                bar -= bar_min_id

                if use_cache:
                    mask = mask_pos.view(b, 1, n)
                    rsm_bar = bar.view(b, 1, n) - bar[:, -1].view(b, 1, 1) + 15
                else:
                    mask = (
                        torch.triu(torch.ones(n, n, device=inp.device), diagonal=1)
                        .bool()
                        .repeat(b, 1, 1)
                    )
                    mask[mask_pos] = True
                    mask[pre_bar.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_bar = bar.view(b, 1, n) - bar.view(b, n, 1) + 15

                rsm_bar[mask] = -1
                rsm_bar[~mask] = torch.clamp(rsm_bar[~mask], min=0, max=15)
                rel_structure_matrix["bar"] = rsm_bar

                pos_min_id = remi_indexer["position_0"]
                pos_max_id = pos_min_id + MAX_POSITION - 1
                pos_mask = (inp >= pos_min_id) & (inp <= pos_max_id)
                pos_tokens_arange = torch.where(pos_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                pos_tokens_idx = torch.cummax(pos_tokens_arange, dim=1)[0]
                pos = torch.where(
                    pos_tokens_idx == 0, torch.tensor(0), inp.gather(1, pos_tokens_idx)
                )
                pre_pos = pos == 0
                mask_pos = pre_pos | pad_pos
                pos -= pos_min_id
                if use_cache:
                    mask = mask_pos.view(b, 1, n)
                    rsm_pos = pos.view(b, 1, n) - pos[:, -1].view(b, 1, 1) + 47
                else:
                    mask = (
                        torch.triu(torch.ones(n, n, device=inp.device), diagonal=1)
                        .bool()
                        .repeat(b, 1, 1)
                    )
                    mask[mask_pos] = True
                    mask[pre_pos.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_pos = pos.view(b, 1, n) - pos.view(b, n, 1) + 47

                rsm_pos[mask] = -1
                rsm_pos[~mask] = torch.clamp(rsm_pos[~mask], min=0, max=94)
                rel_structure_matrix["position"] = rsm_pos
            else:
                bar_min_id = remi_indexer["bar_0"]
                bar_max_id = bar_min_id + MAX_BAR - 1
                bar_mask = (inp >= bar_min_id) & (inp <= bar_max_id)
                bar_tokens_arange = torch.where(bar_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                bar_tokens_idx = torch.cummax(bar_tokens_arange, dim=1)[0]
                bar = torch.where(
                    bar_tokens_idx == 0, torch.tensor(0), inp.gather(1, bar_tokens_idx)
                )
                pre_bar = bar == 0
                mask_bar_pos = pre_bar | pad_pos
                bar -= bar_min_id

                pos_min_id = remi_indexer["position_0"]
                pos_max_id = pos_min_id + MAX_POSITION - 1
                pos_mask = (inp >= pos_min_id) & (inp <= pos_max_id)
                pos_tokens_arange = torch.where(pos_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                pos_tokens_idx = torch.cummax(pos_tokens_arange, dim=1)[0]
                pos = torch.where(
                    pos_tokens_idx == 0, torch.tensor(0), inp.gather(1, pos_tokens_idx)
                )
                pre_pos = pos == 0
                mask_pos_pos = pre_pos | pad_pos
                pos -= pos_min_id

                time = bar * 48 + pos
                if use_cache:
                    mask = mask_bar_pos.view(b, 1, n) | mask_pos_pos.view(b, 1, n)
                    rsm_time = time.view(b, 1, n) - time[:, -1].view(b, 1, 1) + 767
                else:
                    mask = (
                        torch.triu(torch.ones(n, n, device=inp.device), diagonal=1)
                        .bool()
                        .repeat(b, 1, 1)
                    )
                    mask[mask_bar_pos] = True
                    mask[mask_pos_pos] = True
                    mask[pre_bar.view(b, 1, n).repeat(1, n, 1)] = True
                    mask[pre_pos.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_time = time.view(b, 1, n) - time.view(b, n, 1) + 767

                rsm_time[mask] = -1
                rsm_time[~mask] = torch.clamp(rsm_time[~mask], min=0, max=767)
                rel_structure_matrix["time"] = rsm_time

        if rel_pitch:
            if decompose:
                pitch_min_id = remi_indexer["pitch_0"]
                pitch_max_id = pitch_min_id + MAX_PITCH - 1
                pitch_mask = (inp >= pitch_min_id) & (inp <= pitch_max_id)
                pitch_tokens_arange = torch.where(pitch_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                pitch_tokens_idx = torch.cummax(pitch_tokens_arange, dim=1)[0]
                pitch = torch.where(
                    pitch_tokens_idx == 0, torch.tensor(0), inp.gather(1, pitch_tokens_idx)
                )
                pre_pitch = pitch == 0
                mask_pos = pre_pitch | pad_pos
                pitch -= pitch_min_id
                if use_cache:
                    mask = mask_pos.view(b, 1, n)
                    rsm_pitch = pitch.view(b, 1, n) - pitch[:, -1].view(b, 1, 1)
                else:
                    mask = (
                        torch.triu(torch.ones(n, n, device=inp.device), diagonal=1)
                        .bool()
                        .repeat(b, 1, 1)
                    )
                    mask[mask_pos] = True
                    mask[pre_pitch.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_pitch = pitch.view(b, 1, n) - pitch.view(b, n, 1)

                rsm_octave = rsm_pitch // 12 + 11
                rsm_semi = rsm_pitch % 12
                rsm_octave[mask] = -1
                rsm_octave[~mask] = torch.clamp(rsm_octave[~mask], min=0, max=21)
                rsm_semi[mask] = -1
                rel_structure_matrix["octave"] = rsm_octave
                rel_structure_matrix["semitone"] = rsm_semi
            else:
                pitch_min_id = remi_indexer["pitch_0"]
                pitch_max_id = pitch_min_id + MAX_PITCH - 1
                pitch_mask = (inp >= pitch_min_id) & (inp <= pitch_max_id)
                pitch_tokens_arange = torch.where(pitch_mask.to(device), torch.arange(inp.size(1)).to(device), 0)
                pitch_tokens_idx = torch.cummax(pitch_tokens_arange, dim=1)[0]
                pitch = torch.where(
                    pitch_tokens_idx == 0, torch.tensor(0), inp.gather(1, pitch_tokens_idx)
                )
                pre_pitch = pitch == 0
                mask_pos = pre_pitch | pad_pos
                pitch -= pitch_min_id
                if use_cache:
                    mask = mask_pos.view(b, 1, n)
                    rsm_pitch = pitch.view(b, 1, n) - pitch[:, -1].view(b, 1, 1) + 127
                else:
                    mask = (
                        torch.triu(torch.ones(n, n, device=inp.device), diagonal=1)
                        .bool()
                        .repeat(b, 1, 1)
                    )
                    mask[mask_pos] = True
                    mask[pre_pitch.view(b, 1, n).repeat(1, n, 1)] = True
                    rsm_pitch = pitch.view(b, 1, n) - pitch.view(b, n, 1) + 127

                rsm_pitch[mask] = -1
                rsm_pitch[~mask] = torch.clamp(rsm_pitch[~mask], min=0, max=254)
                rel_structure_matrix["pitch"] = rsm_pitch

        return rel_structure_matrix


if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from data_utils import to_prompt
    from dataset import REMIPlusDataset

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    cfg_def = OmegaConf.load("configs/small.yaml")
    cfg_setting = OmegaConf.load(
        "configs/remi_plus/rel_idx_time_pitch_linear_had_a01.yaml"
    )
    cfg = OmegaConf.merge(cfg_def, cfg_setting)

    model = WrapperREMIDecTransformer(cfg.model).to(device)
    model.train()
    print("Model loaded")

    train_dataset = REMIPlusDataset(
        cfg.data.data_root, cfg.data.data_src, "tiny.txt", transpose=True
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    print("Dataset loaded")

    for batch in tqdm(train_loader):
        loss = model(batch.to(device))
        loss.backward()

    print("Training finished")

    test_dataset = REMIPlusDataset(
        cfg.data.data_root, cfg.data.data_src, "tiny.txt", transpose=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    print("Dataset loaded")

    for batch in tqdm(test_loader):
        prompt = to_prompt(batch[0], "REMI+", 5)
        prompt = prompt.unsqueeze(0).to(device) 
        out = model.generate(prompt, sampling_method="top_p")

    print("Generation finished")
