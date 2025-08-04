#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-GPU evaluation script (DDP version).
依赖:
    pip install lpips pytorch-fid omegaconf Pillow scikit-image tqdm
"""

import os, sys, argparse, importlib, yaml, math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from copy import deepcopy

try:
    import torch_npu            # 兼容 Ascend NPU
except Exception:
    torch_npu = None

# ────────────────────────────────────────────────────────────────────────────────
# 一些第三方包
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import lpips
from metrics.inception import InceptionV3

# 项目内模型
from src.WeTok.models.lfqgan import VQModel as WeTok

# ────────────────────────────────────────────────────────────────────────────────
MODEL_TYPE = {
    "WeTok": WeTok,
}

# ────────────────────────────────────────────────────────────────────────────────
def init_distributed():
    """
    单机多卡 NCCL 初始化：
    - 必须在 init_process_group 之前 set_device
    - local_rank 从环境变量读取
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank

def is_main_process():
    return dist.get_rank() == 0

def all_gather_np(array: np.ndarray):
    """把任意 shape 的 numpy array gather 到 rank0，返回 rank0拼接后的 ndarray。"""
    local_size = torch.tensor([array.shape[0]], device="cuda")
    size_list  = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(v.item()) for v in size_list]

    # flatten 传输，最后再 reshape
    flat = torch.from_numpy(array).cuda()
    # 先 gather 尺寸最大的 tensor，其他 pad
    max_size = max(sizes) * array.shape[1]
    padding  = max_size - flat.numel()
    flat_padded = torch.cat([flat.flatten(), torch.zeros(padding, device="cuda")], dim=0)

    gather_list = [torch.empty_like(flat_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, flat_padded)

    if is_main_process():
        chunks = []
        for g, sz in zip(gather_list, sizes):
            chunks.append(g[: sz * array.shape[1]].view(sz, array.shape[1]).cpu().numpy())
        return np.concatenate(chunks, axis=0)
    return None

# ────────────────────────────────────────────────────────────────────────────────
def load_config(path):
    return OmegaConf.load(path)

def load_model(cfg, model_type, ckpt, stage2=False):
    model = MODEL_TYPE[model_type](**cfg.model.init_args)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        if stage2:
            sd_new = deepcopy(sd)
            for k, v in sd.items(): 
                if "encoder" in k:
                    new_k = "model_ema." + k.replace('.', '')
                    sd_new[new_k] = v
            sd = sd_new
        model.load_state_dict(sd, strict=True)
    return model.eval()

# ────────────────────────────────────────────────────────────────────────────────
class VQWrapper(torch.nn.Module):
    """把 encode→decode 封装成 forward，适配 DDP。"""
    def __init__(self, model, model_type):
        super().__init__()
        self.g = model
        self.model_type = model_type

    def forward(self, x):
        if getattr(self.g, "use_ema", False):
            ctx = self.g.ema_scope()
        else:
            ctx = torch.no_grad()  # dummy

        with ctx:
            if self.model_type == "WeTok":  # WeTok
                quant, _, indices, _ = self.g.encode(x)
            recon = self.g.decode(quant)
        return recon, indices

# ────────────────────────────────────────────────────────────────────────────────
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# ────────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True)
    p.add_argument('--ckpt_path',   required=True)
    p.add_argument('--image_size',  type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--workers',     type=int, default=4)
    p.add_argument('--model',       choices=['WeTok'], required=True)
    p.add_argument('--stage2',      type=bool, default=False)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    local_rank = init_distributed()              # 先初始化分布式
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # 1. config & dataset
    cfg = load_config(args.config_file)
    cfg.data.init_args.validation.params.config.size = args.image_size
    cfg.data.init_args.batch_size = args.batch_size

    dataset_conf = cfg.data
    dataset = importlib.import_module(
        dataset_conf["class_path"].rsplit(".", 1)[0]
    ).__getattribute__(
        dataset_conf["class_path"].rsplit(".", 1)[1]
    )(**dataset_conf.init_args)
    dataset.prepare_data()
    dataset.setup()
    val_set = dataset.val_dataloader().dataset   # Dataset 实例
    sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
    loader  = DataLoader(
        val_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    # 2. model & metrics
    base_model = load_model(cfg, args.model, args.ckpt_path, args.stage2)
    wrapper    = VQWrapper(base_model, args.model).to(device)
    model      = torch.nn.parallel.DistributedDataParallel(
        wrapper, device_ids=[device], output_device=device, broadcast_buffers=False
    )

    # inception / lpips 也做 DDP，只是 forward 不梯度
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device).eval()

    lpips_alex = lpips.LPIPS(net='alex').to(device).eval()
    lpips_vgg  = lpips.LPIPS(net='vgg').to(device).eval()

    # 3. 统计量
    codebook_size = cfg.model.init_args.n_embed
    usage_tensor  = torch.zeros(codebook_size, dtype=torch.long, device=device)
    ssim_value = torch.tensor(0.0, device=device)
    psnr_value = torch.tensor(0.0, device=device)
    lpips_alex_sum = torch.tensor(0.0, device=device)
    lpips_vgg_sum = torch.tensor(0.0, device=device)
    img_cnt = torch.tensor(0, dtype=torch.long, device=device)
    iter_cnt = torch.tensor(0, dtype=torch.long, device=device)

    feats_ref, feats_rec = [], []

    # 4. 推理
    with torch.no_grad():
        for batch in tqdm(loader, disable=not is_main_process()):
            images = batch["image"].permute(0, 3, 1, 2).to(device)  # B×3×H×W
            img_cnt += images.size(0)

            rec, indices = model(images)               # rec(-1~1), indices(B, H', W')
            rec = rec.clamp(-1, 1)

            # codebook 使用率
            usage_tensor.index_add_(0, indices.view(-1), torch.ones_like(indices.view(-1)))

            # LPIPS
            lpips_alex_sum += lpips_alex(images, rec).sum()
            lpips_vgg_sum  += lpips_vgg (images, rec).sum()

            # FID features
            feats_ref.append(inception((images + 1) / 2)[0].squeeze(-1).squeeze(-1).cpu().numpy())
            feats_rec.append(inception((rec    + 1) / 2)[0].squeeze(-1).squeeze(-1).cpu().numpy())

            # PSNR / SSIM
            img_np = ((images + 1) / 2).mul_(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rec_np = ((rec    + 1) / 2).mul_(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            ssim_sum = torch.tensor(0.0, device=device)
            psnr_sum = torch.tensor(0.0, device=device)
            for a, b in zip(img_np, rec_np):
                ssim_sum  += ssim_loss(a, b, data_range=255, channel_axis=-1)
                psnr_sum  += psnr_loss(a, b)
            ssim_value += ssim_sum / images.size(0)
            psnr_value += psnr_sum / images.size(0)
            iter_cnt += 1

    # 5. gather
    lpips_t = torch.stack([lpips_alex_sum, lpips_vgg_sum, ssim_value, psnr_value, img_cnt.float(), iter_cnt.float()])
    dist.reduce(lpips_t, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(usage_tensor, dst=0, op=dist.ReduceOp.SUM)

    feats_ref = all_gather_np(np.concatenate(feats_ref, axis=0))
    feats_rec = all_gather_np(np.concatenate(feats_rec, axis=0))

    if is_main_process():
        lpips_alex_sum, lpips_vgg_sum, ssim_value, psnr_value, img_cnt, iter_cnt = lpips_t.tolist()
        lpips_alex_val = lpips_alex_sum / img_cnt
        lpips_vgg_val  = lpips_vgg_sum  / img_cnt
        ssim_val       = ssim_value       / iter_cnt
        psnr_val       = psnr_value       / iter_cnt

        mu1, mu2 = feats_ref.mean(0), feats_rec.mean(0)
        sigma1   = np.cov(feats_ref, rowvar=False)
        sigma2   = np.cov(feats_rec, rowvar=False)
        fid_val  = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        utilization = (usage_tensor > 0).sum().item() / codebook_size

        print("\n=========  Final Results  =========")
        print(f"FID            : {fid_val:.4f}")
        print(f"LPIPS (alex)   : {lpips_alex_val:.4f}")
        print(f"LPIPS (vgg)    : {lpips_vgg_val :.4f}")
        print(f"SSIM           : {ssim_val      :.4f}")
        print(f"PSNR           : {psnr_val      :.4f}")
        print(f"Codebook util. : {utilization*100:.2f}%")
    dist.barrier()

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()