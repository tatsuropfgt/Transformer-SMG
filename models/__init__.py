import torch
from torch import nn

from .note_transformers import WrapperNoteDecTransformer
from .proll_transformers import WrapperPianoRollDecTransformer
from .remi_transformers import WrapperREMIDecTransformer


def init_model(cfg, music_rep, device, use_dp):
    if music_rep == "NoteTuple":
        model = WrapperNoteDecTransformer(cfg)
    elif music_rep == "PianoRoll":
        model = WrapperPianoRollDecTransformer(cfg)
    elif music_rep == "REMI+":
        model = WrapperREMIDecTransformer(cfg)
    else:
        raise NotImplementedError(f"Music representation {music_rep} not implemented")

    model.to(device)
    dp = False
    if torch.cuda.device_count() > 1 and use_dp:
        model = nn.DataParallel(model)
        dp = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    return model, dp
