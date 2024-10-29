import torch.nn as nn


class ForHook(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        return None
