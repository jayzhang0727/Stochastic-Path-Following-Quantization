import torch
import torch.nn.functional as F

def stochastic_msq(step_size, x, boundary_idx, lamb=None):
    '''
    First version of stochastic msq without regularizer.
    The quantization result is clipped to be within boundary idx.

    Parameters
    ----------
    step_size: float
        The step size of the alphabet
    x: float
        The value to be quantized
    boundary_idx: int
        The max idx of the alphebt to not go over
    lamb: dummy variable, not used

    Returns
    -------
    float of the result of msq
    '''
    # stochastic quantization
    p = 1 - x / step_size + torch.floor(x / step_size)  # probability
    prob_mask = torch.bernoulli(p).bool()  
    x[prob_mask] = step_size * torch.floor(x[prob_mask] / step_size) 
    x[~prob_mask] = step_size * (torch.floor(x[~prob_mask] / step_size) + 1)

    # clipping large values
    clipping_mask = (torch.abs(x) > step_size * boundary_idx)
    x[clipping_mask] = torch.sign(x[clipping_mask]) * step_size * boundary_idx
    return x


def msq(step_size, x, boundary_idx, lamb=None):
    '''
    Assuming the alphebt is uniform and symmetric, perform msq

    Parameters
    ----------
    step_size: float
        The step size of the alphabet
    x: float
        The value to be quantized
    boundary_idx: int
        The max idx of the alphebt to not go over
    lamb: dummy variable, not used

    Returns
    -------
    float of the result of msq
    '''
    return torch.sign(x) * step_size * torch.minimum(torch.abs(torch.floor(x / step_size + 0.5)), torch.ones_like(x) * boundary_idx)


def hard_thresholding_msq(step_size, x, boundary_idx, lamb):
    '''
    Hard thresholding quantizer.

    Parameters
    ----------
    step_size: float
        The step size of the alphabet
    x: float
        The value to be quantized
    boundary_idx: int
        The max idx of the alphebt to not go over
    lamb: float
        The boundary for threasholding

    Returns
    -------
    Floating value result of hard thresholding
    '''
    x = F.threshold(torch.abs(x), lamb, 0) * torch.sign(x)  # hard thresholding 
    y = torch.sign(x) * torch.maximum(torch.abs(x)-lamb, torch.zeros_like(x))  # soft thresholding 
    round_val = torch.minimum(torch.abs(torch.floor(y / step_size + 0.5)), torch.ones_like(y) * boundary_idx)
    return torch.sign(x) * (lamb + step_size * round_val) * (torch.abs(x) > lamb).float()


def soft_thresholding_msq(step_size, x, boundary_idx, lamb):
    '''
    Soft thresholding quantizer.

    Parameters
    ----------
    step_size: float
        The step size of the alphabet
    x: float
        The value to be quantized
    boundary_idx: int
        The max idx of the alphebt to not go over
    lamb: float
        The boundary for threasholding

    Returns
    -------
    Floating value result of hard thresholding
    '''
    x = torch.sign(x) * torch.maximum(torch.abs(x)-lamb, torch.zeros_like(x))  # soft thresholding 
    return torch.sign(x) * step_size * torch.minimum(torch.abs(torch.floor(x / step_size + 0.5)), torch.ones_like(x) * boundary_idx)
