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



def sample_ctx_tgt_test(img: torch.Tensor, image_size: int, ctx_type: str):
    """
    Test-time split where context + target = all pixels, no overlap.

    img: (B, N, C), N = image_size**2
    Returns:
        model_kwargs with:
          pos    : positions of TARGET pixels only
          ctx_pos: positions of context pixels
          ctx_x  : values of context pixels
        ctx_idxs: indices of context pixels (in flattened image)
    """
    device = img.device
    B, N, C = img.shape
    assert N == image_size ** 2

    # ----- 1) coords for all pixels -----
    pixel_idx = torch.arange(N, device=device)
    x1, x2 = pixel_idx // image_size, pixel_idx % image_size
    pos = torch.stack(
        [
            2 * x1.float() / (image_size - 1) - 1,  # row in [-1, 1]
            2 * x2.float() / (image_size - 1) - 1,  # col in [-1, 1]
        ],
        dim=-1,
    )[None].repeat(B, 1, 1)  # (B, N, 2)

    # ----- 2) choose context indices + ctx_pos/ctx_x -----
    #if ctx_type == "None":  # no ctx
    #    ctx_idxs = torch.arange(0, 0, device=device).unsqueeze(0).repeat(B, 1)  # (B, 0)
    #    ctx_pos = pos[:, :0]          # (B, 0, 2)
    #    ctx_x = img[:, :0]            # (B, 0, C)

    if ctx_type == "half":  # half image ctx
        base = torch.arange(0, image_size ** 2 // 2, device=device)  # (Nc,)
        ctx_idxs = base.unsqueeze(0).repeat(B, 1)                     # (B, Nc)
        ctx_pos = pos[:, base]                                        # (B, Nc, 2)
        ctx_x = img[:, base]                                          # (B, Nc, C)

    elif ctx_type == "frame":  # outer frame, width=image_size/8
        ctx_upper = torch.arange(0, image_size ** 2 // 8, device=device)
        ctx_lower = torch.arange(7 * image_size ** 2 // 8, image_size ** 2, device=device)

        ctx_left = [
            torch.arange(
                i + image_size ** 2 // 8,
                7 * image_size ** 2 // 8,
                image_size,
                device=device,
            )
            for i in range(image_size // 8)
        ]
        ctx_right = [
            torch.arange(
                image_size ** 2 // 8 + image_size - 1 - i,
                7 * image_size ** 2 // 8,
                image_size,
                device=device,
            )
            for i in range(image_size // 8)
        ]

        base = torch.concat([ctx_upper, ctx_lower] + ctx_left + ctx_right)  # (Nc,)
        ctx_idxs = base.unsqueeze(0).repeat(B, 1)                            # (B, Nc)
        ctx_pos = pos[:, base]                                              # (B, Nc, 2)
        ctx_x = img[:, base]                                                # (B, Nc, C)

    elif ctx_type == "quart":  # quarter image ctx (top + bottom quarters)
        ctx_idxs1 = torch.arange(0, image_size ** 2 // 4, device=device)
        ctx_idxs2 = torch.arange(3 * image_size ** 2 // 4, image_size ** 2, device=device)
        base = torch.concat([ctx_idxs1, ctx_idxs2])                         # (Nc,)
        ctx_idxs = base.unsqueeze(0).repeat(B, 1)                           # (B, Nc)
        ctx_pos = pos[:, base]                                             # (B, Nc, 2)
        ctx_x = img[:, base]                                               # (B, Nc, C)

    else:
        # random ~5% of pixels as context, per image
        ctx_size = image_size ** 2 // 20
        # random permutation of indices per batch element
        rand = torch.rand(B, N, device=device)
        ctx_idxs = rand.argsort(dim=-1)[..., :ctx_size]                    # (B, ctx_size)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)            # (B, 1)
        ctx_pos = pos[batch_idx, ctx_idxs]                                 # (B, ctx_size, 2)
        ctx_x = img[batch_idx, ctx_idxs]                                   # (B, ctx_size, C)

    # ----- 3) compute target indices = complement of ctx_idxs -----
    # mask[i, j] = True iff pixel j is TARGET for batch i
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    if ctx_idxs.numel() > 0:
        mask.scatter_(1, ctx_idxs, False)

    # nonzero returns (batch_idx, col_idx); we want col_idx reshaped per batch
    _, tgt_flat = mask.nonzero(as_tuple=True)          # (B * Ntgt,)
    tgt_idxs = tgt_flat.view(B, -1)                    # (B, Ntgt)

    # positions for target pixels only
    batch_idx = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
    pos_tgt = pos[batch_idx, tgt_idxs]                       # (B, Ntgt, 2)

    # ----- 4) build model kwargs with TARGET positions only -----
    model_kwargs = dict(
        pos=pos_tgt,   # <--- changed: only target positions
        ctx_pos=ctx_pos,
        ctx_x=ctx_x,
    )

    return model_kwargs, tgt_idxs.shape[1]


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
    T=1000,
    ctx_type="half",
    deterministic=False
):

    model.eval()
    # Pull one batch for shape; JAX also takes shapes from current dataset. :contentReference[oaicite:10]{index=10}
    img_size = 64
    B, img_flat_size, C = img.shape

    # batch
    model_kwargs, Ntgt = sample_ctx_tgt_test(img, img_size, ctx_type=ctx_type)
    x = torch.randn(B, Ntgt, C, device=device)
    model_kwargs['dt'] = torch.Tensor([1/T] * x.shape[0]).to(device)

    # loop to denoise
    for t in range(T):
        with torch.no_grad():
            pred = model(x, torch.Tensor([t/T] * x.shape[0]).to(device), **model_kwargs)
        if deterministic:
            alpha = 1.
            sigma = 0.
        else: # stochastic
            alpha = 1 + t / T * (1 - t / T)
            sigma = 0.2 * (t / T * (1 - t / T)) ** 0.5
        x += (alpha * pred + sigma * torch.randn_like(x).to(x.device)) / T
        x = x.clamp_(-1., 1.)

    x = x.detach()

    imgs = build_ctx_tgt_viz_images(
        img_flat=img,  # (B, N, C) in [-1,1]
        ctx_x=model_kwargs['ctx_pos'],
        ctx_y=model_kwargs['ctx_x'],
        tgt_x=model_kwargs['pos'],
        pred_y=x,
        img_size=img_size,
        n_examples=8,  # number of columns
        )

    grid = vutils.make_grid(
        imgs,
        nrow=B,  # == n_examples → 3 rows
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

