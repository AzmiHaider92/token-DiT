import math
import torch
import torch.distributed as dist
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
import wandb
import tqdm
import os
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def sample_ctx_tgt_test(img_flat: torch.Tensor, img_size: int, ctx_frac: float = 0.1):
    """
    Test-time split where context + target = all pixels, no overlap.

    img_flat: (B, N, C), N = img_size**2
    """
    device = img_flat.device
    B, N, C = img_flat.shape
    assert N == img_size**2

    # coords for all pixels, same as before
    pixel_idx = torch.arange(N, device=device)
    x1 = pixel_idx // img_size
    x2 = pixel_idx % img_size
    pos = torch.stack([
        2.0 * x1.float() / (img_size - 1) - 1.0,
        2.0 * x2.float() / (img_size - 1) - 1.0,
    ], dim=-1)          # (N, 2)
    pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    # choose context size
    ctx_size = int(N * ctx_frac)

    # random permutation per image
    idxs = torch.rand(B, N, device=device).argsort(dim=-1)  # (B, N)

    ctx_idxs = idxs[..., :ctx_size]      # (B, ctx_size)

    # target = complement of context
    all_idxs = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
    mask = torch.ones_like(all_idxs, dtype=torch.bool)
    mask.scatter_(1, ctx_idxs, False)
    tgt_idxs = all_idxs[mask].view(B, N - ctx_size)  # (B, N - ctx_size)

    batch_idx = torch.arange(B, device=device).unsqueeze(-1)

    ctx_pos = pos[batch_idx, ctx_idxs]         # (B, ctx_size, 2)
    ctx_x = img_flat[batch_idx, ctx_idxs]    # (B, ctx_size, C)

    tgt_pos = pos[batch_idx, tgt_idxs]         # (B, N - ctx_size, 2)
    tgt_x = img_flat[batch_idx, tgt_idxs]    # if you want GT for eval
    model_kwargs = dict(pos=tgt_pos, ctx_pos=ctx_pos, ctx_x=ctx_x)

    return model_kwargs, tgt_x


def _coords_to_indices(coords: torch.Tensor, img_size: int):
    """
    coords: (..., 2) in [-1,1], order [y, x]
    returns: (y_idx, x_idx) in [0, img_size-1] as long tensors
    """
    # map [-1,1] -> [0, img_size-1]
    y = (coords[..., 0] + 1.0) * 0.5 * (img_size - 1)
    x = (coords[..., 1] + 1.0) * 0.5 * (img_size - 1)

    y_idx = y.round().long().clamp(0, img_size - 1)
    x_idx = x.round().long().clamp(0, img_size - 1)
    return y_idx, x_idx


def build_ctx_tgt_viz_images(
    img_flat: torch.Tensor,
    ctx_x: torch.Tensor,
    ctx_y: torch.Tensor,
    tgt_x: torch.Tensor,
    pred_y: torch.Tensor,   # y1_pred, same thing
    img_size: int,
    n_examples: int = 4,
):
    """
    Build an image tensor suitable for vutils.make_grid to show:

      Row 1: original images
      Row 2: context-only (masked) images
      Row 3: context + predicted targets

    Args:
        img_flat: (B, N, C), N = img_size**2, values in [-1, 1]
        ctx_x:    (B, ctx_size, 2)   normalized coords in [-1,1]^2
        ctx_y:    (B, ctx_size, C)
        tgt_x:    (B, tgt_size, 2)
        pred_y:   (B, tgt_size, C)
        img_size: H = W
        n_examples: how many batch items (columns) to visualize

    Returns:
        imgs: (3 * n_examples, C, H, W) tensor ready for make_grid
              order is:
                [orig_1..orig_N, ctx_1..ctx_N, ctx+pred_1..ctx+pred_N]
              so with nrow=n_examples you'll get 3 rows.
    """
    B, N, C = img_flat.shape
    assert N == img_size ** 2, "img_flat second dim must be img_size**2"
    n_examples = min(n_examples, B)

    orig_list = []
    ctx_list = []
    pred_list = []

    for b in range(n_examples):
        # ----- original (H, W, C) -----
        img_b = img_flat[b].reshape(img_size, img_size, C)  # (H, W, C)

        # ----- context-only image -----
        ctx_img = torch.ones_like(img_b)  # white canvas in [-1,1]
        ctx_y_idx, ctx_x_idx = _coords_to_indices(ctx_x[b], img_size)
        ctx_img[ctx_y_idx, ctx_x_idx] = ctx_y[b]

        # ----- context + predictions image -----
        # start from context-only canvas, then paint predicted targets
        pred_img = ctx_img.clone()
        tgt_y_idx, tgt_x_idx = _coords_to_indices(tgt_x[b], img_size)
        pred_img[tgt_y_idx, tgt_x_idx] = pred_y[b]

        # convert to (C, H, W)
        orig_list.append(img_b.permute(2, 0, 1))
        ctx_list.append(ctx_img.permute(2, 0, 1))
        pred_list.append(pred_img.permute(2, 0, 1))

    # order: all originals, then all ctx, then all ctx+pred
    imgs = torch.stack(orig_list + ctx_list + pred_list, dim=0)  # (3*n_examples, C, H, W)
    return imgs


fid_image_size = 256
@torch.no_grad()
def validate(
    model,
    img,
    device,
    savedir,
    step,
    denoising_steps=1000
):
    # Pull one batch for shape; JAX also takes shapes from current dataset. :contentReference[oaicite:10]{index=10}
    img_size = 64
    B = img.shape[0]

    # batch
    model_kwargs, tgt_x = sample_ctx_tgt_test(
        img_flat=img,
        img_size=img_size,
    )

    x_t = torch.randn(tgt_x.shape, dtype=tgt_x.dtype, device=device)

    dt = 1 / denoising_steps
    # loop to denoise
    with torch.no_grad():
        for ti in range(denoising_steps):
            t = ti / denoising_steps
            t = torch.full((B,), t, device=device, dtype=torch.float32)
            v = model(x_t, t, **model_kwargs)
            x_t = x_t + v * dt

        x1_pred = x_t.detach().clone()

    imgs = build_ctx_tgt_viz_images(
        img_flat=img,  # (B, N, C) in [-1,1]
        ctx_x=model_kwargs['ctx_pos'],
        ctx_y=model_kwargs['ctx_x'],
        tgt_x=model_kwargs['pos'],
        pred_y=x1_pred,
        img_size=64,
        n_examples=8,  # number of columns
        )


    grid = vutils.make_grid(
        imgs,
        nrow=8,  # == n_examples → 3 rows
        padding=2,
        normalize=True,
        value_range=(-1, 1)
    )

    save_path = os.path.join(
        savedir,
        f"pred_epoch{step}.png"
    )
    save_image(grid, save_path)
    wandb.log({"Ctx/Tgt Predictions": wandb.Image(grid)})

