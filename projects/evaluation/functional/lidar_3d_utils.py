import numpy as np
import scipy.linalg
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Tuple

from scipy import linalg

def cdist_rbf(p: torch.Tensor, q: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """RBF kernel"""
    assert p.ndim == q.ndim == 2
    dist = torch.cdist(p, q, p=2.0)
    gamma = 1 / (2 * sigma**2)
    dist = torch.exp(-gamma * dist**2)
    return dist

@torch.no_grad()
def compute_jsd_2d(hist1: torch.Tensor, hist2: torch.Tensor) -> float:
    """BEV-based Jensen-Shannon divergence (JSD)"""
    hist1 = hist1.flatten(1)  # (N, size**2)
    hist2 = hist2.flatten(1)  # (N, size**2)
    p = hist1.sum(dim=0) / hist1.sum()
    q = hist2.sum(dim=0) / hist2.sum()
    jsd = jensenshannon(p.cpu().numpy(), q.cpu().numpy())
    return jsd


@torch.no_grad()
def compute_mmd_2d(hist1: torch.Tensor, hist2: torch.Tensor) -> float:
    """BEV-based maximum mean discrepancy (MMD)"""
    hist1 = hist1.flatten(1)  # (N, size**2)
    hist2 = hist2.flatten(1)  # (N, size**2)
    p = hist1 / hist1.sum(dim=1, keepdim=True)
    q = hist2 / hist2.sum(dim=1, keepdim=True)
    mmd = cdist_rbf(p, p).mean() + cdist_rbf(q, q).mean() - 2 * cdist_rbf(p, q).mean()
    return mmd.item()

def compute_frechet_distance(feats1, feats2):
    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    distance = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return float(distance)

def compute_squared_mmd(feats1, feats2, num_subsets=100, max_subset_size=1000):
    # feats1: (N1, D), feats2: (N1, D)
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _ in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    distance = t / num_subsets / m
    return float(distance)

def point_cloud_to_histogram(
    point_cloud: torch.Tensor,
    field_size: float = 160.0,
    bins: int = 100,
    min_depth: float = 3.0,  # from lidargen
    max_depth: float = 70.0,  # from lidargen
) -> torch.Tensor:
    assert point_cloud.ndim == 2, "must be (N, 3)"
    assert point_cloud.shape[1] == 3, "must be (N, 3)"
    assert bins % 2 == 0
    depth = point_cloud.norm(p=2, dim=1)
    mask = (depth > min_depth) & (depth < max_depth)
    bound = field_size / 2
    hist = torch.histogramdd(
        point_cloud[mask, 0:2].cpu(),  # xy
        bins=bins,
        range=[-bound, bound, -bound, bound],
    ).hist
    return hist

class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.amax(dim=2)  # [B,1024]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(-1, 3, 3)
        x = x + torch.eye(3, device=x.device)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.amax(dim=2)  # [B,1024]
        if self.global_feat:
            return x, trans
        else:
            _, _, N = pointfeat.shape
            x = x[..., None].repeat_interleave(N, dim=-1)
            return torch.cat([x, pointfeat], 1), trans


class PointNet1(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, trans = self.feat(x)
        x2 = F.relu(self.bn1(self.fc1(x1)))
        x3 = F.relu(self.bn2(self.fc2(x2)))
        x4 = self.fc3(x3)
        feature = torch.cat((x1, x2, x3, x4), dim=1)
        return feature


def pretrained_pointnet(
    dataset: str = "shapenet", device: torch.device = "cpu", compile: bool = True
) -> nn.Module:
    if dataset == "shapenet":
        model = PointNet1(k=16)
        state_dict = load_state_dict_from_url(
            url="https://github.com/microsoft/SpareNet/raw/main/Frechet/cls_model_39.pth",
            progress=True,
            weights_only=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False)
    if compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model.to(device)