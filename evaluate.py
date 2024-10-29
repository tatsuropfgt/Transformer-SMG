import argparse

from omegaconf import OmegaConf

from main import settings_to_fn
from train import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--sampling_method", type=str, default="top_p")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.size == "small":
        cfg_def = OmegaConf.load("configs/small.yaml")
    else:
        raise ValueError(f"invalid size: {args.size}")

    cfg_setting = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)

    fn, _ = settings_to_fn(cfg, args)

    train_config = TrainConfig(cfg, fn, eval_only=True)
    train_config.evaluate(args.sampling_method, args.threshold)


if __name__ == "__main__":
    main()
