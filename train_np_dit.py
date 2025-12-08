# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))

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

    assert args.global_batch_size % world == 0, f"Batch size must be divisible by world size."
    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/{args.expname}*"))
        experiment_index_str = f"{experiment_index:03d}"  if experiment_index > 0 else ""
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{args.expname}{experiment_index_str}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank)

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

    diffusion = create_diffusion(timestep_respacing="", flow_matching=args.flow_matching)  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = 1e-4
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

    # ----- wandb -----
    if (not is_ddp) or rank == 0:
        # timestamped run name
        ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
        run_name = f"Token-DiT_{ts}"

        wandb.init(
            project="Generative_sampling",
            name=run_name,
            id=None,
            mode='online'
        )

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for img, _ in loader:
            local_batch_size = img.shape[0]
            x1, x2 = pixel_idx//args.image_size, pixel_idx%args.image_size
            pos = torch.stack([2*x1.float()/(args.image_size-1) - 1, 
                             2*x2.float()/(args.image_size-1) - 1], -1).to(device)[None].repeat(local_batch_size, 1, 1)
            x = img.to(device).reshape(local_batch_size, args.num_channels, args.image_size*args.image_size).transpose(1, 2)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
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
            model_kwargs = dict(pos=tgt_pos, ctx_pos=ctx_pos, ctx_x=ctx_x)
            loss_dict = diffusion.training_losses(model, tgt_x, t, model_kwargs)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
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
                             denoising_steps=4)

            # Save DiT checkpoint:
            #if train_steps == args.ckpt_every//10 or train_steps == args.ckpt_every//6  or train_steps == args.ckpt_every//2 or (train_steps % args.ckpt_every == 0 and train_steps > 0):
            #    if rank == 0:
            #        checkpoint = {
            #            "model": model.module.state_dict(),
            #            "ema": ema.state_dict(),
            #            "opt": opt.state_dict(),
            #            "args": args
            #        }
            #        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            #        torch.save(checkpoint, checkpoint_path)
            #        logger.info(f"Saved checkpoint to {checkpoint_path}")
            #dist.barrier()

    #model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--flow-matching", type=bool, default=False, help="Use velocity prediction.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume training from.")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--num-channels", type=int, choices=[3, 1], default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=-1)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
