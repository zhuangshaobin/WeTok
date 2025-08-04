"""
We provide Tokenizer Evaluation code here.
Following Cosmos, we use original resolution to evaluate
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
https://github.com/NVIDIA/Cosmos-Tokenizer/blob/main/cosmos_tokenizer/image_lib.py

This script is modified to support Distributed Data Parallel (DDP) for multi-GPU evaluation.
"""

import os
import sys
sys.path.append(os.getcwd())
import torch
try:
    import torch_npu
except: 
    pass

# DDP related imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from src.WeTok.models.lfqgan import VQModel as WeTok
from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse

# DDP: Define a function to setup distributed environment
def setup_ddp():
    """Initializes the distributed process group."""
    # torchrun will set these environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # It is recommended to use nccl for NVIDIA GPUs
    dist.init_process_group(backend="nccl")
    
    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

## for different model configuration
MODEL_TYPE = {
    "WeTok": WeTok,
}

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
    model = MODEL_TYPE[model_type](**config.model.init_args) 
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # It's good practice to print missing/unexpected keys, especially in DDP
        # to ensure all ranks load the same model.
        if dist.get_rank() == 0:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
    return model.eval()

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def pad_images(batch, spatial_align = 16):
    height, width = batch.shape[2:4]
    align = spatial_align
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0
    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = torch.nn.functional.pad(batch, (width_to_pad >> 1,  width_to_pad - (width_to_pad >> 1), height_to_pad >> 1, height_to_pad - (height_to_pad >> 1), 0, 0, 0, 0), "constant", 0)
    return batch, crop_region

def unpad_images(batch, crop_region):
    assert len(crop_region) == 4, "crop_region should be len of 4."
    y1, x1, y2, x2 = crop_region
    return batch[:, :, y1:y2, x1:x2]

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--original_reso", action="store_true")
    parser.add_argument("--model", choices=["WeTok"])
    # DDP: Add workers argument for DataLoader
    parser.add_argument("--workers", default=8, type=int, help="Number of workers for DataLoader")
    return parser.parse_args()

def main(args):
    # DDP: Setup distributed environment
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    config_data = OmegaConf.load(args.config_file)
    # DDP: Each GPU processes a batch, so the effective batch size is args.batch_size * world_size
    config_data.data.init_args.batch_size = args.batch_size
    config_data.data.init_args.validation.params.config.original_reso = args.original_reso

    config_model = load_config(args.config_file, display=(rank == 0))
    # DDP: Load model to the specific device for this process
    model = load_vqgan_new(config_model, model_type=args.model, ckpt_path=args.ckpt_path).to(device)
    # DDP: Wrap the model with DistributedDataParallel
    # find_unused_parameters can be helpful for some models, but usually not needed for inference
    model = DDP(model, device_ids=[local_rank])
    
    codebook_size = config_model.model.init_args.n_embed
    
    usage = {i: 0 for i in range(codebook_size)}

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    
    # DDP: Use DistributedSampler to distribute data across GPUs
    # We need to get the underlying dataset from the dataloader provided by the custom dataset object
    val_dataset = dataset._val_dataloader().dataset
    sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # DDP: Create a new DataLoader with the sampler
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False # Important to not drop last samples for correct evaluation
    )

    pred_xs = []
    pred_recs = []
    ssim_value = 0.0
    psnr_value = 0.0
    num_iter = 0

    # DDP: Only show tqdm progress bar on the main process (rank 0)
    if rank == 0:
        pbar = tqdm(val_dataloader)
    else:
        pbar = val_dataloader

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].permute(0, 3, 1, 2).to(device)
            images, crop_region = pad_images(images)
            if images.shape[-1] > 2000 or images.shape[-2] > 2000:
                continue
            
            # DDP: Access the original model through `model.module`
            original_model = model.module
            if original_model.use_ema:
                with original_model.ema_scope():
                    if args.model == "WeTok":
                        quant, diff, indices, _ = original_model.encode(images)
                    reconstructed_images = original_model.decode(quant)
            else:
                if args.model == "WeTok":
                    quant, diff, indices, _ = original_model.encode(images)
                reconstructed_images = original_model.decode(quant)

            reconstructed_images = reconstructed_images.clamp(-1, 1)
            reconstructed_images = unpad_images(reconstructed_images, crop_region)
            images = unpad_images(images, crop_region)
            
            for index in indices:
                usage[index.item()] += 1
            
            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            pred_x = inception_model(images)[0].squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0].squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            
            ssim_temp = 0
            psnr_temp = 0
            B = rgb_restored.shape[0]
            for i in range(B):
                ssim_temp += ssim_loss(rgb_restored[i], rgb_gt[i], data_range=1.0, channel_axis=-1)
                psnr_temp += psnr_loss(rgb_gt[i], rgb_restored[i])
            
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1

    # DDP: All processes must wait here until all other processes are done.
    dist.barrier()

    # --- DDP: Aggregation Step ---
    # Gather all results from all processes to the main process (rank 0)
    
    # 1. Gather scalar values (SSIM, PSNR, num_iter)
    metrics_tensor = torch.tensor([ssim_value, psnr_value, num_iter], dtype=torch.float64).to(device)
    dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

    # 2. Gather list of numpy arrays (for FID) and dict (for usage) using all_gather_object
    # This is a convenient way to gather arbitrary python objects.
    gathered_xs_lists = [None] * world_size
    gathered_recs_lists = [None] * world_size
    gathered_usage_dicts = [None] * world_size
    
    dist.all_gather_object(gathered_xs_lists, pred_xs)
    dist.all_gather_object(gathered_recs_lists, pred_recs)
    dist.all_gather_object(gathered_usage_dicts, usage)

    # DDP: Only rank 0 performs the final calculation and printing
    if rank == 0:
        # Process gathered FID features
        flat_xs = [item for sublist in gathered_xs_lists for item in sublist]
        flat_recs = [item for sublist in gathered_recs_lists for item in sublist]
        
        pred_xs_total = np.concatenate(flat_xs, axis=0)
        pred_recs_total = np.concatenate(flat_recs, axis=0)

        mu_x = np.mean(pred_xs_total, axis=0)
        sigma_x = np.cov(pred_xs_total, rowvar=False)
        mu_rec = np.mean(pred_recs_total, axis=0)
        sigma_rec = np.cov(pred_recs_total, rowvar=False)

        fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
        
        # Process gathered scalar metrics
        total_ssim = metrics_tensor[0].item()
        total_psnr = metrics_tensor[1].item()
        total_iters = metrics_tensor[2].item()
        
        final_ssim = total_ssim / total_iters
        final_psnr = total_psnr / total_iters

        # Process gathered usage dictionaries
        final_usage = {i: 0 for i in range(codebook_size)}
        for u_dict in gathered_usage_dicts:
            for key, value in u_dict.items():
                final_usage[key] += value
        
        num_count = sum([1 for value in final_usage.values() if value > 0])
        utilization = num_count / codebook_size

        print("\n--- Final Results ---")
        print(f"Total images evaluated: {len(pred_xs_total)}")
        print(f"FID: {fid_value}")
        print(f"SSIM: {final_ssim}")
        print(f"PSNR: {final_psnr}")
        print(f"Codebook Utilization: {utilization}")
  
    # DDP: Clean up the process group
    cleanup_ddp()

if __name__ == "__main__":
    args = get_args()
    main(args)