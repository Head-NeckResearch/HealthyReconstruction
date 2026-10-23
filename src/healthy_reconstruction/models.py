"""Study network definitions and denoising corruption, without fitted weights."""
from __future__ import annotations
import argparse
import math
import random
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def compute_norm(N: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = M.astype(bool)
    mean, std = [], []
    for c in range(3):
        vals = N[..., c][valid]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mean.append(0.0); std.append(1.0)
        else:
            mean.append(float(vals.mean()))
            s = float(vals.std())
            std.append(s if s > 1e-8 else 1.0)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


class StaticNeutralAE(nn.Module):
    def __init__(self, n_markers: int = 105, hidden: int = 256, latent: int = 48, dropout: float = 0.20):
        super().__init__()
        self.n_markers = int(n_markers)
        in_dim = n_markers * 4
        out_dim = n_markers * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.GELU(),
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, N, _ = n.shape
        x = torch.cat([n, m.unsqueeze(-1)], dim=-1).reshape(B, N * 4)
        y = self.net(x).reshape(B, N, 3)
        return y


def corrupt_static(n: torch.Tensor, m: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = n.shape
    keep = m > 0.5
    rand_marker_drop = torch.rand(B, N, device=n.device) < args.marker_mask_prob
    keep = keep & (~rand_marker_drop)
    # Whole-half random drop encourages inference from partial contralateral/central face.
    if args.half_mask_prob > 0:
        for b in range(B):
            if torch.rand((), device=n.device) < args.half_mask_prob:
                if torch.rand((), device=n.device) < 0.5:
                    keep[b, : N // 2] = False
                else:
                    keep[b, N // 2 :] = False
    noisy = n + torch.randn_like(n) * args.noise_std
    noisy = torch.where(keep.unsqueeze(-1), noisy, torch.zeros_like(noisy))
    return noisy, keep.float()

class DynamicMotionProjector(nn.Module):
    def __init__(self, n_markers:int=105, n_frames:int=100, hidden:int=160, latent:int=40, dropout:float=0.20):
        super().__init__(); self.n_markers=n_markers; self.n_frames=n_frames; self.in_dim=n_markers*4; self.out_dim=n_markers*3
        self.enc1=nn.Sequential(nn.Conv1d(self.in_dim,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc2=nn.Sequential(nn.Conv1d(hidden,hidden,5,stride=2,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc3=nn.Sequential(nn.Conv1d(hidden,latent,5,stride=2,padding=2),nn.BatchNorm1d(latent),nn.GELU(),nn.Dropout(dropout))
        self.dec1=nn.Sequential(nn.Conv1d(latent,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.dec2=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.out=nn.Conv1d(hidden,self.out_dim,3,padding=1)
    def forward(self,x:torch.Tensor,m:torch.Tensor)->torch.Tensor:
        B,T,N,_=x.shape
        y=torch.cat([x,m.unsqueeze(-1)],dim=-1).reshape(B,T,N*4).permute(0,2,1)
        y=self.enc1(y); y=self.enc2(y); y=self.enc3(y)
        y=F.interpolate(y,size=max(1,math.ceil(T/2)),mode='linear',align_corners=False); y=self.dec1(y)
        y=F.interpolate(y,size=T,mode='linear',align_corners=False); y=self.dec2(y); y=self.out(y)
        return y.permute(0,2,1).reshape(B,T,N,3)


def corrupt_dynamic(x: torch.Tensor, m: torch.Tensor, args: argparse.Namespace):
    B,T,N,_=x.shape; keep=m>0.5
    keep = keep & (torch.rand(B,T,N,device=x.device) > args.point_mask_prob)
    marker_drop=torch.rand(B,N,device=x.device)<args.marker_mask_prob
    keep = keep & (~marker_drop[:,None,:])
    # temporal block masks
    for b in range(B):
        for _ in range(args.n_time_blocks):
            if torch.rand((),device=x.device)<args.time_block_prob:
                L=int(torch.randint(args.time_block_min,args.time_block_max+1,(1,),device=x.device).item())
                s=int(torch.randint(0,max(1,T-L+1),(1,),device=x.device).item())
                keep[b,s:s+L,:]=False
    xin=x + torch.randn_like(x)*args.noise_std
    xin=torch.where(keep.unsqueeze(-1),xin,torch.zeros_like(xin))
    return xin, keep.float()


def normalize(values, mask, mean, std):
    """Per-coordinate optimizer-fitted normalization; unsupported inputs are zero."""
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, bool)
    z = (values-np.asarray(mean,np.float32))/np.asarray(std,np.float32)
    return np.where(mask[...,None] & np.isfinite(z), z, 0.).astype(np.float32)


def masked_error_sums(pred, target, mask):
    """Sanitize unsupported NaNs before subtraction to preserve finite gradients."""
    supported = mask.unsqueeze(-1).bool()
    squared = (torch.where(supported,pred,0.)-torch.where(supported,target,0.)).square()
    return squared.sum(), mask.sum()*target.shape[-1]
