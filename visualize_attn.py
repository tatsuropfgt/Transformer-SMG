import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str)
    
    args = parser.parse_args()

    attn_values = np.load(args.npy_path)
    file_name = args.npy_path.split("/")[-1].split(".")[0]

    cmap = sns.color_palette("viridis", as_cmap=True)
    # fig, axes = plt.subplots(4, 1, figsize=(24, 16))
    # for i in range(4):
    #     ax = axes[i]
    #     sns.heatmap(attn_values[i,:,48:73], ax=ax, cmap=cmap, cbar=True, vmin=-1.5, vmax=1.5)
    #     ax.set_title(f"layer {i+1}", fontsize=15)
    #     ax.set_ylabel("attention heads", fontsize=12)
    #     # ax.set_xticks(np.arange(0, 121, 12) + 0.5)
    #     # ax.set_xticklabels(range(-60, 61, 12), rotation=0)
    #     ax.set_xticks(np.arange(0, 25) + 0.5)
    #     ax.set_xticklabels(["-12", "", "", "", "", "", "", "", "", "", "", "", "0", "", "", "", "", "", "", "", "", "", "", "", "12"], rotation=0)
    #     ax.set_yticks([])

    plt.figure(figsize=(24, 4))
    sns.heatmap(attn_values[:, :, 48:73].mean(axis=1), cmap=cmap, cbar=True, vmin=-0.5, vmax=1)
    plt.title(f"{file_name}", fontsize=15)
    # plt.ylabel("layer", fontsize=12)
    # plt.xlabel("pitch", fontsize=12)
    plt.xticks(np.arange(0, 25) + 0.5, list(range(-12, 13)), rotation=0, fontsize=16)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"figs/{file_name}.png")

if __name__ == "__main__":
    main()
