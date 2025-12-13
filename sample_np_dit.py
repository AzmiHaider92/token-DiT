# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from datetime import datetime

import torch
from tqdm import tqdm

from train_np_dit import ddp_setup, print_gpu_info, set_seed, maybe_wrap_ddp
from vis import sample_ctx_tgt_test, build_ctx_tgt_viz_images

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from latent_token_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image, make_grid


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
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Load a checkpoint if specified:
    checkpoint = torch.load(args.ckpt, weights_only=False)
    train_args = checkpoint["args"]

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
    run_name = f"Sampling_{args.expname}T{args.timesteps}__{ts}"

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = f"{args.results_dir}/{run_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

    # broadcast dir names so all ranks have the same strings (even though only 0 uses them)
    if is_ddp:
        dir_list = [experiment_dir, checkpoint_dir]
        dist.broadcast_object_list(dir_list, src=0)
        experiment_dir, checkpoint_dir = dir_list

    # Create model:
    model = DiT_models[train_args.model](
        num_classes=train_args.num_classes,
        in_channels=train_args.num_channels,
        num_tokens=train_args.image_size**2,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    requires_grad(model, False)  # Enable gradients for training
    model = maybe_wrap_ddp(model, device, is_ddp)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    ema.load_state_dict(checkpoint["ema"], strict=False)
    requires_grad(ema, False)

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

    model.eval()  
    ema.eval()

    n = args.global_batch_size
    w = args.image_size
    c = args.num_channels

    img = next(iter(loader))[0].to(device) # Get a batch of images
    local_batch_size = img.shape[0]
    img = img.to(device).reshape(local_batch_size, args.num_channels, args.image_size * args.image_size).transpose(1, 2)
    model_kwargs, Ntgt = sample_ctx_tgt_test(img, w, "random5")

    all_noisy_samples = []
    T = args.timesteps
    print(f"T={T}")
    model_kwargs['dt'] = torch.Tensor([1/T] * n).to(device)
    print(f"Generating stochastic samples")
    x = torch.randn(n, Ntgt, c, device=device)
    # loop to denoise
    for t in tqdm(range(T)):
        with torch.no_grad():
            pred = model(x, torch.Tensor([t / T] * x.shape[0]).to(device), **model_kwargs)
        alpha = 1 + t / T * (1 - t / T)
        sigma = 0.2 * (t / T * (1 - t / T)) ** 0.5
        x += (alpha * pred + sigma * torch.randn_like(x).to(x.device)) / T
        x = x.clamp_(-1., 1.)

    print("Done generating samples.")
    x = x.detach()

    imgs = build_ctx_tgt_viz_images(
        img_flat=img,  # (B, N, C) in [-1,1]
        ctx_x=model_kwargs['ctx_pos'],
        ctx_y=model_kwargs['ctx_x'],
        tgt_x=model_kwargs['pos'],
        pred_y=x,
        img_size=args.image_size,
        n_examples=8,  # number of columns
    )
    grid = vutils.make_grid(
        imgs,
        nrow=n,  # == n_examples → 3 rows
        padding=2,
        normalize=True,
        value_range=(-1, 1)
    )

    save_path = f"{experiment_dir}/stochastic_samples.png"
    save_image(grid, save_path)



if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--num-channels", type=int, choices=[3, 1], default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--ctx_type", type=str, choices=["none", "half", "quart", "frame", "random"], default="random")
    args = parser.parse_args()
    main(args)
