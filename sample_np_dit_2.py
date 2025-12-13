# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sampling + visualization script:
- Visualize multiple batches (viz_batches)
- For each batch: show 1x (original, context-only) + one prediction row per T in {1,2,10,100} (or user-provided)
Grid layout (per saved PNG):
  Row 1: Original
  Row 2: Context-only
  Row 3..: Context + prediction for each T
Columns: n_examples
"""

from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
import argparse
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from train_np_dit import ddp_setup, print_gpu_info, set_seed, maybe_wrap_ddp
from vis import sample_ctx_tgt_test  # your function
from latent_token_models import DiT_models

# speed flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Helper Functions                                  #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


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


def build_multiT_viz_grid(
    img_flat: torch.Tensor,              # (B, N, C) in [-1,1]
    ctx_pos: torch.Tensor,               # (B, Nc, 2) coords
    ctx_val: torch.Tensor,               # (B, Nc, C) values
    tgt_pos: torch.Tensor,               # (B, Ntgt, 2) coords
    preds_by_T: list,                    # [(T, pred_val(B,Ntgt,C)), ...] in desired order
    img_size: int,
    n_examples: int = 8,
):
    """
    Returns imgs tensor shaped ((2 + len(Ts)) * n_examples, C, H, W) for make_grid with nrow=n_examples:
      Row 1: original
      Row 2: context-only
      Row 3..: context + prediction for each T
    """
    B, N, C = img_flat.shape
    assert N == img_size ** 2
    n_examples = min(n_examples, B)
    H = W = img_size

    orig_list = []
    ctx_list = []

    # Build orig + ctx rows ONCE
    ctx_hwcs = []  # keep (H,W,C) for fast cloning per T
    for b in range(n_examples):
        img_b = img_flat[b].reshape(H, W, C)  # (H,W,C)

        ctx_img = torch.ones_like(img_b)      # white canvas in [-1,1]
        if ctx_pos.numel() > 0 and ctx_pos.shape[1] > 0:
            cy, cx = _coords_to_indices(ctx_pos[b], H)
            ctx_img[cy, cx] = ctx_val[b]

        orig_list.append(img_b.permute(2, 0, 1))  # (C,H,W)
        ctx_list.append(ctx_img.permute(2, 0, 1))
        ctx_hwcs.append(ctx_img)                  # (H,W,C)

    # Build prediction rows per T
    pred_rows = []
    for (T, pred_val) in preds_by_T:
        pred_list = []
        for b in range(n_examples):
            pred_img = ctx_hwcs[b].clone()  # (H,W,C)
            if tgt_pos.numel() > 0 and tgt_pos.shape[1] > 0:
                ty, tx = _coords_to_indices(tgt_pos[b], H)
                pred_img[ty, tx] = pred_val[b]  # (Ntgt,C) broadcast to pixels via ty/tx
            pred_list.append(pred_img.permute(2, 0, 1))  # (C,H,W)
        pred_rows.append(torch.stack(pred_list, dim=0))  # (n_examples,C,H,W)

    orig_imgs = torch.stack(orig_list, dim=0)  # (n_examples,C,H,W)
    ctx_imgs = torch.stack(ctx_list, dim=0)    # (n_examples,C,H,W)

    # Stack rows: orig, ctx, pred(T=...), pred(T=...), ...
    imgs = torch.cat([orig_imgs, ctx_imgs] + pred_rows, dim=0)
    return imgs


def parse_T_list(s: str):
    # e.g. "1,2,10,100"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    Ts = [int(p) for p in parts]
    assert all(t >= 1 for t in Ts), "All T must be >= 1"
    return Ts


#################################################################################
#                                  Main                                         #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    checkpoint = torch.load(args.ckpt, weights_only=False)
    train_args = checkpoint["args"]

    # Setup DDP
    is_ddp, device, rank, world, local_rank = ddp_setup()
    print_gpu_info(rank, world, local_rank)
    set_seed(args.global_seed)

    # Experiment dir (only rank 0 creates)
    experiment_dir = None
    checkpoint_dir = None
    ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
    run_name = f"Sampling_{args.expname}__Ts_{args.T_list.replace(',', '_')}__{ts}"

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = f"{args.results_dir}/{run_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        create_logger(experiment_dir)

    if is_ddp:
        dir_list = [experiment_dir, checkpoint_dir]
        dist.broadcast_object_list(dir_list, src=0)
        experiment_dir, checkpoint_dir = dir_list

    # Create model
    model = DiT_models[train_args.model](
        num_classes=train_args.num_classes,
        in_channels=train_args.num_channels,
        num_tokens=train_args.image_size ** 2,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    requires_grad(model, False)
    model = maybe_wrap_ddp(model, device, is_ddp)
    model.eval()

    # (ema not used here, but keep consistent with checkpoint)
    ema = deepcopy(model).to(device)
    ema.load_state_dict(checkpoint["ema"], strict=False)
    requires_grad(ema, False)
    ema.eval()

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
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

    # IMPORTANT: all ranks should run forwards if model is DDP (to avoid hangs from buffer broadcasts)
    # Only rank 0 will save images.
    loader_it = iter(loader)

    for b_i in range(args.viz_batches):
        batch = next(loader_it)[0].to(device)  # (B_local, C, H, W)
        B_local = batch.shape[0]
        H = W = args.image_size
        C = args.num_channels

        # flatten to (B, N, C) in [-1,1]
        img = batch.reshape(B_local, C, H * W).transpose(1, 2)

        # ctx/tgt split
        model_kwargs, Ntgt = sample_ctx_tgt_test(
            img=img,
            image_size=args.image_size,
            ctx_type=args.ctx_type,
            rand_perc=args.ctx_rand_perc
        )

        preds_by_T = []

        for T in T_list:
            # If shortcut expects dt, set dt per sample
            if getattr(train_args, "use_shortcut", False):
                model_kwargs["dt"] = torch.full((B_local,), 1.0 / float(T), device=device)

            # init targets
            x = torch.randn(B_local, Ntgt, C, device=device)

            # integrate
            for t in range(T):
                tt = torch.full((B_local,), float(t) / float(T), device=device)
                with torch.no_grad():
                    pred = model(x, tt, **model_kwargs)

                alpha = 1.0
                sigma = 0.0
                x = x + (alpha * pred + sigma * torch.randn_like(x)) / float(T)
                x = x.clamp_(-1., 1.)

            preds_by_T.append((T, x.detach()))

        # Build + save grid (rank 0 only)
        if rank == 0:
            n_examples = min(args.viz_n_examples, B_local)

            imgs = build_multiT_viz_grid(
                img_flat=img,
                ctx_pos=model_kwargs["ctx_pos"],  # coords
                ctx_val=model_kwargs["ctx_x"],    # values
                tgt_pos=model_kwargs["pos"],      # target coords
                preds_by_T=preds_by_T,            # [(T, pred)]
                img_size=args.image_size,
                n_examples=n_examples
            )

            grid = vutils.make_grid(
                imgs,
                nrow=n_examples,          # columns = n_examples
                padding=2,
                normalize=True,
                value_range=(-1, 1),
            )

            save_path = f"{experiment_dir}/samples_b{b_i}_ctx{args.ctx_type}_Ts_{'_'.join(map(str, T_list))}.png"
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
    parser.add_argument("--viz-batches", type=int, default=4, help="How many batches to visualize (and save PNGs for).")
    parser.add_argument("--viz-n-examples", type=int, default=8, help="How many columns (examples) per grid.")
    parser.add_argument("--T-list", type=str, default="1,2,10,100", help='Comma-separated list, e.g. "1,2,10,100".')

    args = parser.parse_args()
    main(args)
