import torch
import torch.nn.functional as F
from functools import partial

torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version[0] > 1 or torch_version[0] == 1 and torch_version[1] > 2:
    grid_sample = partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample

def charbonier_loss(delta, alpha: float=0.45, epsilon: float=1e-3):
    if delta.numel() == 0:
        return 0
    return (delta.pow(2) + epsilon*epsilon).pow(alpha).mean()

def warp_images_with_flow(images, flow):
    N, C, H, W = images.size()
    assert flow.size()[0] == N, f'Number of images and flows should be the same {N} vs {flow.size()[0]}'
    assert flow.size()[1] == 2, f'Flow should contain 2 channels (dx and dy)'
    assert flow.size()[2] == H, f'Height of images and flows should be the same {H} vs {flow.size()[2]}'
    assert flow.size()[3] == W, f'Width of images and flows should be the same {W} vs {flow.size()[3]}'
    # H, W
    device = images.device
    grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                          torch.arange(W, dtype=torch.float32, device=device))
    # H, W, 2 <-- (x, y)
    grid = torch.cat(tuple(map(lambda x: x.unsqueeze(2), grid[::-1])), dim=2)
    # N, H, W, 2
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    # N, H, W, 2
    flow = flow.permute(0, 2, 3, 1)
    grid = grid + flow
    # normalize [0, W-1] -> [0, 2]
    grid[..., 0] /= (W-1)/2.
    # normalize [0, H-1] -> [0, 2]
    grid[..., 1] /= (H-1)/2.
    # normalize [0, 2] -> [-1, 1]
    grid -= 1
    return grid_sample(images, grid)

def photometric_loss(prev_images, next_images, flow):
    total_photometric_loss = 0.
    looss_weight_sum = 0.

    warped = warp_images_with_flow(next_images, flow)
    return charbonier_loss(warped - prev_images)

def smoothness_loss(flow):
    ucrop = flow[..., 1:, :]
    dcrop = flow[..., :-1, :]
    lcrop = flow[..., 1:]
    rcrop = flow[..., :-1]

    ulcrop = flow[..., 1:, 1:]
    drcrop = flow[..., :-1, :-1]
    dlcrop = flow[..., :-1, 1:]
    urcrop = flow[..., 1:, :-1]

    return (charbonier_loss(lcrop - rcrop) + charbonier_loss(ucrop - dcrop) +\
            charbonier_loss(ulcrop - drcrop) + charbonier_loss(dlcrop - urcrop)) / 4
