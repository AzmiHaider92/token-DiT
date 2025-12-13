# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sampling + visualization script (PREDICT-ALL-PIXELS):
- Model is conditioned on context (ctx_pos, ctx_x)
- BUT model predicts ALL pixels (pos = full grid), so the generated image is prediction-only.
Grid layout (per saved PNG):
  Row 1: Original (GT)
  Row 2: Context-only (white canvas + GT context pixels)
  Row 3..: Prediction-only for each T in T_list
Columns: n_examples
"""

from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from train_np_dit import ddp_setup, print_gpu_info, set_seed, maybe_wrap_ddp
from latent_token_models import DiT_models

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Helper Functions                                  #
#################################################################################

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    # From ADM
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _coords_to_indices(coords: torch.Tensor, img_size: int):
    """
    coords: (..., 2) in [-1,1], order [y, x]
    returns: (y_idx, x_idx) in [0, img_size-1] long tensors
    """
    if coords.numel() == 0:
        empty = torch.empty(coords.shape[:-1], dtype=torch.long, device=coords.device)
        return empty, empty

    y = (coords[..., 0] + 1.0) * 0.5 * (img_size - 1)
    x = (coords[..., 1] + 1.0) * 0.5 * (img_size - 1)
    y_idx = y.round().long().clamp(0, img_size - 1)
    x_idx = x.round().long().clamp(0, img_size - 1)
    return y_idx, x_idx


def parse_T_list(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    Ts = [int(p) for p in parts]
    assert all(t >= 1 for t in Ts), "All T must be >= 1"
    return Ts


@torch.no_grad()
def sample_ctx_allpos_test(img: torch.Tensor, image_size: int, ctx_type: str, rand_perc: float):
    """
    Test-time split:
      - Context: some pixels (ctx_pos + ctx_x)
      - Query points (pos): ALL pixels, so model predicts all pixels.

    img: (B, N, C), N=image_size**2 in row-major order
    Returns:
        model_kwargs: dict with
            pos     : (B, N, 2) coords for ALL pixels (query)
            ctx_pos : (B, Nc, 2) coords for context pixels
            ctx_x   : (B, Nc, C) values for context pixels
        N: int (=image_size**2)
    """
    device = img.device
    B, N, C = img.shape
    assert N == image_size ** 2

    # coords for all pixels (row-major)
    pixel_idx = torch.arange(N, device=device)
    y, x = pixel_idx // image_size, pixel_idx % image_size
    pos_all = torch.stack(
        [
            2 * y.float() / (image_size - 1) - 1,  # y in [-1,1]
            2 * x.float() / (image_size - 1) - 1,  # x in [-1,1]
        ],
        dim=-1,
    )[None].expand(B, -1, -1)  # (B, N, 2)

    ctx_type_l = (ctx_type or "").lower()

    if ctx_type_l == "none":
        ctx_pos = pos_all[:, :0]
        ctx_x = img[:, :0]

    elif ctx_type_l == "half":
        base = torch.arange(0, N // 2, device=device)
        ctx_pos = pos_all[:, base]
        ctx_x = img[:, base]

    elif ctx_type_l == "quart":
        base = torch.cat([
            torch.arange(0, N // 4, device=device),
            torch.arange(3 * N // 4, N, device=device),
        ])
        ctx_pos = pos_all[:, base]
        ctx_x = img[:, base]

    elif ctx_type_l == "frame":
        w = max(1, image_size // 8)

        ctx_upper = torch.arange(0, image_size * w, device=device)
        ctx_lower = torch.arange(N - image_size * w, N, device=device)

        ctx_left = [
            torch.arange(image_size * w + i, N - image_size * w, image_size, device=device)
            for i in range(w)
        ]
        ctx_right = [
            torch.arange(image_size * w + (image_size - 1 - i), N - image_size * w, image_size, device=device)
            for i in range(w)
        ]

        base = torch.cat([ctx_upper, ctx_lower] + ctx_left + ctx_right)
        base = torch.unique(base)  # safety
        ctx_pos = pos_all[:, base]
        ctx_x = img[:, base]

    else:
        # random context per image
        ctx_size = int(rand_perc * N)
        ctx_size = max(0, min(ctx_size, N))
        if ctx_size == 0:
            ctx_pos = pos_all[:, :0]
            ctx_x = img[:, :0]
        else:
            r = torch.rand(B, N, device=device)
            ctx_idxs = r.topk(k=ctx_size, dim=-1, largest=False).indices  # (B, ctx_size)
            bidx = torch.arange(B, device=device).unsqueeze(1)
            ctx_pos = pos_all[bidx, ctx_idxs]
            ctx_x = img[bidx, ctx_idxs]

    model_kwargs = dict(
        pos=pos_all,     # <--- ALL pixels queried
        ctx_pos=ctx_pos,
        ctx_x=ctx_x,
    )
    return model_kwargs, N


def build_multiT_viz_grid_predonly(
    img_flat: torch.Tensor,      # (B, N, C) in [-1,1]
    ctx_pos: torch.Tensor,       # (B, Nc, 2)
    ctx_x: torch.Tensor,         # (B, Nc, C)
    preds_by_T: list,            # [(T, pred(B,N,C)), ...]
    img_size: int,
    n_examples: int = 8,
):
    """
    Rows:
      1) Original (GT)
      2) Context-only (white + GT context pixels)
      3..) Prediction-only for each T (reshaped from pred(B,N,C))

    Returns imgs: ((2+len(Ts))*n_examples, C, H, W) for make_grid(nrow=n_examples)
    """
    B, N, C = img_flat.shape
    assert N == img_size ** 2
    n_examples = min(n_examples, B)
    H = W = img_size

    orig_list = []
    ctx_list = []

    # orig + ctx rows once
    for b in range(n_examples):
        img_b = img_flat[b].reshape(H, W, C)  # (H,W,C)

        ctx_img = torch.ones_like(img_b)      # white
        if ctx_pos.numel() > 0 and ctx_pos.shape[1] > 0:
            cy, cx = _coords_to_indices(ctx_pos[b], H)
            ctx_img[cy, cx] = ctx_x[b]

        orig_list.append(img_b.permute(2, 0, 1))
        ctx_list.append(ctx_img.permute(2, 0, 1))

    # prediction-only rows per T
    pred_rows = []
    for (T, pred_val) in preds_by_T:
        pred_list = []
        for b in range(n_examples):
            pred_img = pred_val[b].reshape(H, W, C)  # <-- prediction only, no GT overlay
            pred_list.append(pred_img.permute(2, 0, 1))
        pred_rows.append(torch.stack(pred_list, dim=0))

    orig_imgs = torch.stack(orig_list, dim=0)  # (n_examples,C,H,W)
    ctx_imgs = torch.stack(ctx_list, dim=0)    # (n_examples,C,H,W)

    imgs = torch.cat([orig_imgs, ctx_imgs] + pred_rows, dim=0)
    return imgs


#################################################################################
#                                  Main                                         #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."

    checkpoint = torch.load(args.ckpt, weights_only=False)
    train_args = checkpoint["args"]

    # DDP
    is_ddp, device, rank, world, local_rank = ddp_setup()
    print_gpu_info(rank, world, local_rank)
    set_seed(args.global_seed)

    # experiment dir
    experiment_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
        run_name = f"SamplingAllPix_{args.expname}__Ts_{args.T_list.replace(',', '_')}__{ts}"
        experiment_dir = f"{args.results_dir}/{run_name}"
        os.makedirs(experiment_dir, exist_ok=True)

    if is_ddp:
        dir_list = [experiment_dir]
        dist.broadcast_object_list(dir_list, src=0)
        experiment_dir = dir_list[0]

    # model
    model = DiT_models[train_args.model](
        num_classes=train_args.num_classes,
        in_channels=train_args.num_channels,
        num_tokens=train_args.image_size ** 2,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    requires_grad(model, False)
    model = maybe_wrap_ddp(model, device, is_ddp)
    model.eval()

    # data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    T_list = parse_T_list(args.T_list)
    loader_it = iter(loader)

    for b_i in range(args.viz_batches):
        batch = next(loader_it)[0].to(device)  # (B_local, C, H, W)
        B_local = batch.shape[0]
        H = W = args.image_size
        C = args.num_channels

        # flatten to (B, N, C)
        img = batch.reshape(B_local, C, H * W).transpose(1, 2)  # (B, N, C)

        # context split BUT query ALL pixels
        model_kwargs, N_all = sample_ctx_allpos_test(
            img=img,
            image_size=args.image_size,
            ctx_type=args.ctx_type,
            rand_perc=args.ctx_rand_perc
        )

        preds_by_T = []

        for T in T_list:
            # If shortcut expects dt, set it per sample
            if getattr(train_args, "use_shortcut", False):
                print("using shortcuts ...")
                model_kwargs["dt"] = torch.full((B_local,), 1.0 / float(T), device=device)

            # x is ALL pixels now
            x = torch.randn(B_local, N_all, C, device=device)

            # integrate
            for t in range(T):
                tt = torch.full((B_local,), float(t) / float(T), device=device)
                with torch.no_grad():
                    pred = model(x, tt, **model_kwargs)  # (B, N, C)

                alpha = 1.0
                sigma = 0.0
                x = x + (alpha * pred + sigma * torch.randn_like(x)) / float(T)
                x = x.clamp_(-1., 1.)

            preds_by_T.append((T, x.detach()))

        # save only on rank0
        if rank == 0:
            n_examples = min(args.viz_n_examples, B_local)

            imgs = build_multiT_viz_grid_predonly(
                img_flat=img,
                ctx_pos=model_kwargs["ctx_pos"],
                ctx_x=model_kwargs["ctx_x"],
                preds_by_T=preds_by_T,
                img_size=args.image_size,
                n_examples=n_examples
            )

            grid = vutils.make_grid(
                imgs,
                nrow=n_examples,
                padding=2,
                normalize=True,
                value_range=(-1, 1),
            )

            save_path = f"{experiment_dir}/samplesAllPix_b{b_i}_ctx{args.ctx_type}_Ts_{'_'.join(map(str, T_list))}.png"
            save_image(grid, save_path)
            print(f"[rank0] Saved: {save_path}")

    if is_ddp:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--num-channels", type=int, choices=[3, 1], default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--ctx_type", type=str, choices=["none", "half", "quart", "frame", "random"], default="random")
    parser.add_argument("--ctx-rand-perc", type=float, default=0.05)

    # VIZ controls
    parser.add_argument("--viz-batches", type=int, default=4)
    parser.add_argument("--viz-n-examples", type=int, default=8)
    parser.add_argument("--T-list", type=str, default="1,2,10,100,1000")

    args = parser.parse_args()
    main(args)
