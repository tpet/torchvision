import torch
import torch.nn.functional as F


def nan_loss(loss, input, target, *args, **kwargs):
    valid = torch.isfinite(target)
    if not valid.any():
        return torch.tensor(float('nan'))
    input = torch.masked_select(input, valid)
    target = torch.masked_select(target, valid)
    return loss(input, target, *args, **kwargs)


def nan_mse_loss(input, target, *args, **kwargs):
    input = input.flatten()
    target = target.flatten()
    return nan_loss(F.mse_loss, input, target, *args, **kwargs)
