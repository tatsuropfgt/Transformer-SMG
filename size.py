from main import settings_to_fn
import argparse
from omegaconf import OmegaConf
import os
from models import init_model

check_configs = [
    "configs/note_tuple/baseline.yaml",
    "configs/note_tuple/rel_idx_a01.yaml",
    "configs/note_tuple/rel_idx_time_pitch_linear_a01.yaml",
    "configs/note_tuple/rel_idx_time_pitch_linear_had_a01.yaml",
    "configs/piano_roll/baseline.yaml",
    "configs/piano_roll/rel_idx_a01.yaml",
    "configs/piano_roll/rel_idx_time_pitch_linear_a01.yaml",
    "configs/piano_roll/rel_idx_time_pitch_linear_had_a01.yaml",
    "configs/remi_plus/baseline.yaml",
    "configs/remi_plus/rel_idx_a01.yaml",
    "configs/remi_plus/rel_idx_time_pitch_linear_a01.yaml",
    "configs/remi_plus/rel_idx_time_pitch_linear_had_a01.yaml",
]

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="small")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

cfg_def = OmegaConf.load("configs/small.yaml")

for config in check_configs:
    cfg_setting = OmegaConf.load(config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)
    fn = settings_to_fn(cfg, args)
    print(fn)
    save_path = os.path.join(cfg.save_dir, fn)
    _, _ = init_model(cfg.model, cfg.music_rep, "cpu", False)
    
