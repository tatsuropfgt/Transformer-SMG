import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import sys
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NoteTupleDataset
from main import settings_to_fn
from models import init_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def check_calculate_norm_attn(kwargs, to_out):
    # h x i x j
    attn = kwargs["attn"][0]
    h = attn.shape[0]
    # h x j x d
    value = kwargs["value"][0]
    # k x k
    W_out = to_out.weight.T
    # k x k -> h x d x k
    W_out = rearrange(W_out, "(h d) k -> h d k", h=h)

    out1 = torch.einsum("h i j, h j d -> h i d", attn, value)
    out1 = rearrange(out1, "h n d -> n (h d)")
    # n x k
    out1 = to_out(out1)

    out2 = torch.einsum("h i j, h j d -> h i j d", attn, value)
    out2 = torch.einsum("h i j d, h d k -> h i j k", out2, W_out)
    out2 = out2.sum(dim=2).sum(dim=0)

    return torch.allclose(out1, out2, atol=0.001)


def calculate_norm_attn(kwargs, to_out):
    # h x i x j
    attn = kwargs["attn"][0]
    h = attn.shape[0]
    # h x j x d
    value = kwargs["value"][0]
    # k x k
    W_out = to_out.weight.T
    # k x k -> h x d x k
    W_out = rearrange(W_out, "(h d) k -> h d k", h=h)
    out = torch.einsum("h i j, h j d -> h i j d", attn, value)
    out = torch.einsum("h i j d, h d k -> h i j k", out, W_out)
    # h x i x j
    out = out.norm(dim=-1)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    cfg_def = OmegaConf.load("configs/small.yaml")
    cfg_setting = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)
    cfg.data.batch_size = 1
    fn, _ = settings_to_fn(cfg, args)

    model, _ = init_model(cfg.model, cfg.music_rep, "cpu", False)
    model = model.to(device)

    class SimpleHook:
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

        def hook_fn(self, module, args, kwargs, output):
            self.args = args
            self.kwargs = kwargs
            self.output = output

    def safe_divide(a, b):
        result = np.zeros_like(a, dtype=a.dtype)
        mask = (b != 0) & (a != 0)
        result[mask] = np.divide(a[mask], b[mask])
        return result

    if args.norm:
        hooks = [
            SimpleHook(model.net.layers[i].attn.normattn_hook)
            for i in range(len(model.net.layers))
        ]
        to_out = [model.net.layers[i].attn.to_out for i in range(len(model.net.layers))]

    else:
        hooks = [
            SimpleHook(model.net.layers[i].attn.attn_hook)
            for i in range(len(model.net.layers))
        ]

    dataset = NoteTupleDataset(
        cfg.data.data_root, cfg.data.data_src, cfg.data.train_file, False
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    layer_num = len(model.net.layers)
    head_num = model.net.layers[0].attn.heads

    # attn_time_sum = np.array(
    #     [[[0.0 for _ in range(768)] for _ in range(head_num)] for _ in range(layer_num)]
    # )  # 0 ~ 767
    # attn_time_num = np.array(
    #     [[[0 for _ in range(768)] for _ in range(head_num)] for _ in range(layer_num)]
    # )  # 0 ~ 767
    attn_pitch_sum = np.array(
        [[[0.0 for _ in range(121)] for _ in range(head_num)] for _ in range(layer_num)]
    )  # -60 ~ 60
    attn_pitch_num = np.array(
        [[[0 for _ in range(121)] for _ in range(head_num)] for _ in range(layer_num)]
    )  # -60 ~ 60

    idx = 0
    with torch.no_grad():
        torch.manual_seed(0)
        for batch in tqdm(loader):
            _ = model(batch.to(device))
            # time
            time = torch.where(
                batch[0, :-1, 1] == -100,
                -100,
                (batch[0, :-1, 1] - 1) * 48 + batch[0, :-1, 2],
            )
            idx_not_100 = torch.where(time != -100)[0]
            # time_rel = time[idx_not_100[-1]] - time
            pitch = batch[0, :-1, 4]
            pitch_rel = pitch[idx_not_100[-1]] - pitch
            idx_query = idx_not_100[-1]

            for layer_idx in range(layer_num):
                # h x n x n
                if args.norm:
                    if not check_calculate_norm_attn(hooks[layer_idx].kwargs, to_out[layer_idx]):
                        raise ValueError("check_calculate_norm_attn failed")

                    attn = calculate_norm_attn(
                        hooks[layer_idx].kwargs, to_out[layer_idx]
                    )
                else:
                    attn = hooks[layer_idx].kwargs["attn"][0]
                # attn = attn.mean(dim=0)
                for head_idx in range(head_num):
                    for idx_key in idx_not_100:
                        # attn_time_sum[layer_idx][head_idx][time_rel[idx_key]] += attn[
                        #     head_idx, idx_query, idx_key
                        # ].item()
                        # attn_time_num[layer_idx][head_idx][time_rel[idx_key]] += 1

                        if pitch_rel[idx_key] >= -60 and pitch_rel[idx_key] <= 60:
                            attn_pitch_sum[layer_idx][head_idx][
                                pitch_rel[idx_key] + 60
                            ] += attn[head_idx, idx_query, idx_key].item()
                            attn_pitch_num[layer_idx][head_idx][
                                pitch_rel[idx_key] + 60
                            ] += 1


    # attn_time_avg = safe_divide(attn_time_sum, attn_time_num)
    # attn_time_avg_64 = attn_time_avg[:, :, ::12]
    # mean = np.mean(attn_time_avg_64, axis=2, keepdims=True)
    # std = np.std(attn_time_avg_64, axis=2, keepdims=True)
    # normalized_attn_time_avg_64 = (attn_time_avg_64 - mean) / std

    # cmap = sns.color_palette("viridis", as_cmap=True)
    # fig, axes = plt.subplots(4, 1, figsize=(32, 16))

    if args.norm:
        norm = "norm_"
    else:
        norm = ""


    # for i in range(4):
    #     ax = axes[i]
    #     sns.heatmap(normalized_attn_time_avg_64[i], ax=ax, cmap=cmap, cbar=True)
    #     ax.set_title(f"layer {i+1}", fontsize=15)
    #     ax.set_ylabel("attention heads", fontsize=12)
    #     ax.set_xticks(np.arange(0, 64, 4) + 0.5)
    #     ax.set_xticklabels(range(16), rotation=0)
    #     ax.set_yticks([])

    # plt.tight_layout()
    # plt.savefig(f"figs/{norm}{fn}_time.png")
    # plt.close()

    attn_pitch_avg = safe_divide(attn_pitch_sum, attn_pitch_num)
    mean = np.mean(attn_pitch_avg, axis=2, keepdims=True)
    std = np.std(attn_pitch_avg, axis=2, keepdims=True)
    normalized_attn_pitch_avg = (attn_pitch_avg - mean) / std

    # np.save(f"npy/{norm}{fn}_time", normalized_attn_time_avg_64)
    np.save(f"npy/{norm}{fn}_pitch", normalized_attn_pitch_avg)

    # cmap = sns.color_palette("viridis", as_cmap=True)
    # fig, axes = plt.subplots(4, 1, figsize=(64, 16))
    # for i in range(4):
    #     ax = axes[i]
    #     sns.heatmap(normalized_attn_pitch_avg[i], ax=ax, cmap=cmap, cbar=True, vmin=-3, vmax=3)
    #     ax.set_title(f"layer {i+1}", fontsize=15)
    #     ax.set_ylabel("attention heads", fontsize=12)
    #     ax.set_xticks(np.arange(0, 121, 12) + 0.5)
    #     ax.set_xticklabels(range(-60, 61, 12), rotation=0)
    #     ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig(f"figs/{norm}{fn}_pitch.png")

    # cmap = sns.color_palette("viridis", as_cmap=True)
    # fig, axes = plt.subplots(4, 1, figsize=(24, 16))
    # for i in range(4):
    #     ax = axes[i]
    #     sns.heatmap(normalized_attn_pitch_avg[i,:,48:73], ax=ax, cmap=cmap, cbar=True, vmin=-1.5, vmax=1.5)
    #     ax.set_title(f"layer {i+1}", fontsize=15)
    #     ax.set_ylabel("attention heads", fontsize=12)
    #     # ax.set_xticks(np.arange(0, 121, 12) + 0.5)
    #     # ax.set_xticklabels(range(-60, 61, 12), rotation=0)
    #     ax.set_xticks(np.arange(0, 25) + 0.5)
    #     ax.set_xticklabels(["-12", "", "", "", "", "", "", "", "", "", "", "", "0", "", "", "", "", "", "", "", "", "", "", "", "12"], rotation=0)
    #     ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig(f"figs/{norm}{fn}_pitch.png")


if __name__ == "__main__":
    main()
