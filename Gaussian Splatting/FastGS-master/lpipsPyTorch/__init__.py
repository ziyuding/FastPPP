import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    '''Function that measures'''
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
