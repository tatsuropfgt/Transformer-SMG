import argparse

import torch
from omegaconf import OmegaConf

from train import TrainConfig

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--evaluate", type=bool, default=True)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def settings_to_fn(cfg, args):
    size = args.size

    music_rep_map = {"NoteTuple": "n", "PianoRoll": "p", "REMI+": "r"}
    pos_map = {"learnable": "l", "sinusoidal": "s", "None": "n"}
    music_rep = music_rep_map[cfg.music_rep]
    pos_enc = pos_map[cfg.model.pos_enc]

    def bool_to_str(x: bool, char):
        return char if x else ""

    rel_idx = bool_to_str(cfg.model.rel_idx, "i")
    rel_time = bool_to_str(cfg.model.rel_time, "t")
    rel_pitch = bool_to_str(cfg.model.rel_pitch, "p")
    linear = bool_to_str(cfg.model.linear, "l")
    hadamard = bool_to_str(cfg.model.hadamard, "h")
    debug = bool_to_str(args.debug, "d")
    srel = bool_to_str(cfg.model.simple_rel, "s") if hasattr(cfg.model, "simple_rel") else ""
    attn = ""
    if hasattr(cfg.model, "attn_only") and cfg.model.attn_only:
        layer_num = cfg.model.depth
        attn = "o" + str(layer_num)
    else:
        cfg.model.attn_only = False

    decompose = ""
    if hasattr(cfg.model, "decompose") and not cfg.model.decompose:
        decompose = "nd"
    else:
        cfg.model.decompose = True

    if "alpha" in cfg.model:
        alpha = cfg.model.alpha
        if alpha == 0.1:
            alpha01 = "a01"
        elif alpha == 1:
            alpha01 = ""
        else:
            raise ValueError(f"invalid alpha: {alpha}")
    else:
        alpha01 = ""
        cfg.model.alpha = 1

    if hasattr(cfg.model, "rel_tp_sinusoidal") and cfg.model.rel_tp_sinusoidal:
        rel_tp = "sin"
    else:
        rel_tp = ""
        cfg.model.rel_tp_sinusoidal = False
    
    if cfg.model.dim != 256:
        dim = str(cfg.model.dim)
    else:
        dim = ""

    return f"{size}_{music_rep}{pos_enc}_{rel_idx}{rel_time}{rel_pitch}{linear}{hadamard}{alpha01}{srel}{attn}{decompose}{rel_tp}{dim}_{debug}", cfg


def main():
    args = parse_args()

    if args.size == "small":
        cfg_def = OmegaConf.load("configs/small.yaml")
    else:
        raise ValueError(f"invalid size: {args.size}")

    cfg_setting = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)

    if args.debug:
        cfg.train.max_steps = 100
        cfg.train.eval_steps = 50

    fn, cfg = settings_to_fn(cfg, args)

    train_config = TrainConfig(cfg, fn)
    train_config.train()

    if args.evaluate:
        train_config.evaluate()


if __name__ == "__main__":
    main()
