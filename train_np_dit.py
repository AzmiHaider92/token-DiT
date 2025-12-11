# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import math

import torch

from vis import validate

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torch import nn
from latent_token_models import DiT_models
from diffusion import create_diffusion
import os
import torch
from datetime import timedelta
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import wandb
from datetime import datetime


def ddp_setup():
    # --- read envs from torchrun / slurm ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(
        f"[ddp_setup] RANK={rank} WORLD_SIZE={world_size} "
        f"LOCAL_RANK={local_rank} is_ddp={is_ddp}",
        flush=True,
    )

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )

    return is_ddp, device, rank, world_size, local_rank


def ddp_mean(loss: torch.Tensor) -> float | None:
    x = loss.detach().float()               # stay on GPU, fp32
    if dist.is_available() and dist.is_initialized():
        dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)   # sum to rank 0
        if dist.get_rank() == 0:
            x /= dist.get_world_size()
            return x.item()
        return None
    return x.item()


def maybe_wrap_ddp(model: nn.Module, device: torch.device, is_ddp: bool):
    if model is None:
        return None
    if not is_ddp:
        return model
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )

def print_gpu_info(rank=0, world_size=1, local_rank=0):
    if rank == 0:  # only rank 0 prints
        print("===================================")
        print(f"🌍 WORLD_SIZE = {world_size}")
        print(f"🖥️  Visible GPUs = {torch.cuda.device_count()}")
    print(f"[Rank {rank}/{world_size}] -> using cuda:{local_rank}")
    print("===================================")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def sample_timesteps(
    T: int,
    shape,
    device,
    use_shortcut: bool,
    shortcut_frac: float = 0.25,
    shortcut_mode: str = "integer",   # "integer" | "dyadic" | "dyadic_full"
):
    """
    Sample (t, dt) for FM + shortcut training, with *normalized* time.

    Conventions:
        - Internal integer indices: 0 ... T-1
        - Exposed t, dt are in [0, 1], as floats.
        - Shortcut is a backward jump: t2 = t - dt  (in normalized units)
        - dt > 0  => shortcut example, with jump length dt
        - dt == 0 => pure FM example (no shortcut)

    Args:
        T: total number of discrete timesteps (integer).
        shape: e.g. x.shape; only shape[0] (batch size) is used.
        device: a torch.device.
        use_shortcut: enable shortcut sampling if True.
        shortcut_frac: fraction of batch that will use shortcuts (0..1).
                       First B_sc examples will be shortcut; rest FM.
        shortcut_mode:
            - "integer":
                dt_idx ∈ {1, ..., T-1}
                t_idx ∈ {dt_idx, ..., T-1}
            - "dyadic":
                dt_idx ∈ {1, 2, 4, ..., 2^k ≤ T-1}
                t_idx is multiple of dt_idx: t_idx ∈ {dt_idx, 2dt_idx, ..., ≤ T-1}
            - "dyadic_full":
                same as "dyadic" but also includes full-path jump dt_idx = T-1.

    Returns:
        t:  FloatTensor of shape (B,) in [0, 1]
        dt: FloatTensor of shape (B,) in [0, 1], dt == 0 for FM, > 0 for shortcut.
    """
    B = shape[0]

    # --- default: FM everywhere ---
    # FM time indices: uniform over {0, ..., T-1}
    t_idx = torch.randint(low=0, high=T, size=(B,), device=device)
    # normalized t in [0,1)
    t = t_idx.float() / float(T)

    # dt = 0 means "no shortcut" / pure FM, in normalized units
    dt = torch.zeros(B, dtype=torch.float32, device=device)

    if (not use_shortcut) or (T <= 1) or (B == 0):
        return t, dt

    # --- shortcut: only a fraction of the batch ---
    B_sc = int(B * float(shortcut_frac))
    if B_sc == 0:
        return t, dt

    # ---------- 1) Build dt_choices (integer indices) for shortcut part ----------
    if shortcut_mode == "integer":
        # dt_idx ∈ {1, ..., T-1}
        dt_choices = torch.arange(1, T, device=device, dtype=torch.long)

    elif shortcut_mode in ("dyadic", "dyadic_full"):
        # dyadic dt_idx: {1, 2, 4, ..., 2^k <= T-1}
        max_pow = int(math.log2(T - 1))  # T > 1 here
        powers = torch.tensor(
            [2 ** k for k in range(max_pow + 1)],
            device=device,
            dtype=torch.long,
        )
        powers = powers[powers <= (T - 1)]

        if shortcut_mode == "dyadic_full":
            # explicitly include full-path jump dt_idx = T-1 as well
            full_jump = torch.tensor([T - 1], device=device, dtype=torch.long)
            dt_choices = torch.unique(torch.cat([powers, full_jump]))
        else:
            dt_choices = powers

    else:
        raise ValueError(f"Unknown shortcut_mode: {shortcut_mode}")

    # safety: ensure dt_choices not empty
    if dt_choices.numel() == 0:
        return t, dt

    # ---------- 2) Sample dt_idx for the shortcut part ----------
    choice_idx = torch.randint(
        low=0,
        high=dt_choices.numel(),
        size=(B_sc,),
        device=device,
    )
    dt_idx_sc = dt_choices[choice_idx]  # (B_sc,)

    # ---------- 3) Sample t_idx for the shortcut part given dt_idx_sc ----------
    if shortcut_mode == "integer":
        # t_idx_sc ∈ {dt_idx_sc, ..., T-1} (contiguous)
        max_offset = T - dt_idx_sc              # >= 1 because dt_idx_sc <= T-1
        u = torch.rand(B_sc, device=device)
        offset = (u * max_offset.float()).floor().long()  # 0 ... max_offset-1
        t_idx_sc = dt_idx_sc + offset

    else:  # "dyadic" or "dyadic_full"
        # t_idx_sc must be multiple of dt_idx_sc: t_idx_sc = m * dt_idx_sc,
        # with m ∈ {1, ..., floor((T-1)/dt_idx_sc)} so t_idx_sc <= T-1
        max_m = (T - 1) // dt_idx_sc           # shape (B_sc,), each >= 1
        u = torch.rand(B_sc, device=device)
        m = 1 + (u * max_m.float()).floor().long()  # 1..max_m[i]
        t_idx_sc = m * dt_idx_sc               # ∈ [dt_idx_sc, T-1]

    # ---------- 4) Convert indices -> normalized times ----------
    t_sc = t_idx_sc.float() / float(T)      # in [0,1)
    dt_sc = dt_idx_sc.float() / float(T)    # in (0,1]

    # ---------- 5) Mix shortcut and FM ----------
    t[:B_sc] = t_sc
    dt[:B_sc] = dt_sc
    # t[B_sc:] stays FM (random in [0,1))
    # dt[B_sc:] stays 0.0

    return t, dt


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    is_ddp, device, rank, world, local_rank = ddp_setup()
    print_gpu_info(rank, world, local_rank)
    set_seed(args.global_seed)

    # ================== experiment dir (only rank 0 creates) ==================
    experiment_dir = None
    checkpoint_dir = None

    # ----- wandb -----
    run = None
    ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
    run_name = f"Token-DiT_{args.expname}_{ts}"

    if (not is_ddp and rank == 0) or (is_ddp and rank == 0):
        run = wandb.init(
            project="Generative_sampling",
            name=run_name,
            id=None,
            mode="online",
        )

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{run_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

    # broadcast dir names so all ranks have the same strings (even though only 0 uses them)
    if is_ddp:
        dir_list = [experiment_dir, checkpoint_dir]
        dist.broadcast_object_list(dir_list, src=0)
        experiment_dir, checkpoint_dir = dir_list

    # now make logger: only rank 0 actually writes
    logger = create_logger(experiment_dir if rank == 0 else None, rank)
    if rank == 0:
        logger.info(f"Experiment directory created at {experiment_dir}")


    # Create model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        in_channels=args.num_channels,
        num_tokens=args.image_size**2,
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = maybe_wrap_ddp(model, device, is_ddp)

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.train_T, # default: 1000 steps, linear noise schedule
                                 flow_matching=args.flow_matching, use_shortcut=args.use_shortcut)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # Prepare models for training:
    update_ema(ema, model.module if is_ddp else model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.resume:
        # Load a checkpoint if specified:
        logger.info(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.module.load_state_dict(checkpoint["model"], strict=False)
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        args_resumed = checkpoint["args"]

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
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
    pixel_idx = torch.arange(args.image_size**2, device=device)  # Used to create coordinates for images
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for img, _ in loader:
            local_batch_size = img.shape[0]
            x1, x2 = pixel_idx//args.image_size, pixel_idx%args.image_size
            pos = torch.stack([2*x1.float()/(args.image_size-1) - 1, 
                             2*x2.float()/(args.image_size-1) - 1], -1).to(device)[None].repeat(local_batch_size, 1, 1)
            x = img.to(device).reshape(local_batch_size, args.num_channels, args.image_size*args.image_size).transpose(1, 2)
            t, dt = sample_timesteps(args.train_T, x.shape, device, args.use_shortcut,
                                     shortcut_mode=args.shortcut_mode, shortcut_frac=args.shortcut_frac)
            # half image training
            # model_kwargs = dict(pos=pos, ctx_pos=pos[:, :pos.shape[1]//2], ctx_x=x[:, :pos.shape[1]//2]) 
            # set context as random set of pixels (different for each sample in batch)
            ctx_size = torch.randint(low=10, high=args.image_size**2//4, size=[1]).item()
            tgt_size = args.image_size**2//3
            idxs = torch.cuda.FloatTensor(local_batch_size, args.image_size**2).uniform_().argsort(-1).to(img.device)
            ctx_idxs = idxs[...,:ctx_size]
            tgt_idxs = idxs[...,ctx_size//2:tgt_size] 
            # get the relevan coordinates and values of the random idxs
            ctx_pos = pos[torch.arange(local_batch_size).unsqueeze(1), ctx_idxs]
            ctx_x = x[torch.arange(local_batch_size).unsqueeze(1), ctx_idxs]
            tgt_pos = pos[torch.arange(local_batch_size).unsqueeze(1), tgt_idxs]
            tgt_x = x[torch.arange(local_batch_size).unsqueeze(1), tgt_idxs]
            model_kwargs = dict(pos=tgt_pos, ctx_pos=ctx_pos, ctx_x=ctx_x, dt=dt)
            loss_dict = diffusion.training_losses(model, tgt_x, t, model_kwargs, sc_frac=args.shortcut_frac)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model.module if is_ddp else model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                avg_loss = ddp_mean(loss)
                if (not is_ddp) or rank == 0:
                    wandb.log({
                        "training/loss": avg_loss,
                        "training/lr": lr,
                    }, step=train_steps)
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT samples:
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Saving DiT samples at step {train_steps}...")
                    # complete images
                    validate(ema,
                             x,
                             device,
                             experiment_dir,
                             train_steps,
                             args.sample_T,
                             args.sample_ctx_type)

            # Save DiT checkpoint:
            if (train_steps % args.ckpt_every == 0 and train_steps > 0):
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict() if is_ddp else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            #dist.barrier()

    #model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict() if is_ddp else model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/final.pt"
        torch.save(checkpoint, checkpoint_path)
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--flow-matching", type=bool, default=False, help="Use velocity prediction.")
    parser.add_argument("--train-T", type=int, default=1000)
    parser.add_argument("--use-shortcut", type=bool, default=False, help="In flow matching, use shortcuts.")
    parser.add_argument("--shortcut-mode", type=str, choices=["integer", "dyadic", "dyadic_full"], default="integer")
    parser.add_argument("--shortcut-frac", type=float, default=0.25)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume training from.")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--num-channels", type=int, choices=[3, 1], default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=-1)
    parser.add_argument("--sample-T", type=int, default=1000)
    parser.add_argument("--sample-ctx-type", type=str, choices=["half", "frame", "quart", "random5"], default="half")

    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
