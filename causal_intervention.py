import os
import argparse

import torch
from torch import nn
from omegaconf import OmegaConf

from main import settings_to_fn
from models import init_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg_def = OmegaConf.load("configs/small.yaml")
cfg_setting = OmegaConf.load("configs/note_tuple/baseline.yaml")
cfg = OmegaConf.merge(cfg_def, cfg_setting)
cfg.data.batch_size = 1

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="small")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

fn, _ = settings_to_fn(cfg, args)
save_path = os.path.join(cfg.save_dir, fn, "model_best.pt")

model, _ = init_model(cfg.model, cfg.music_rep, device, False)
model.load_state_dict(torch.load(save_path))
model.eval()

print(model)

# TODO
# Ben さんの手法適用してサブスペースの観測
# SAE 学習して何を捉えているかを人手で解釈

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

for 