from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from dsca.losses import ssim_loss


def pairwise_cosine_distance(query: Tensor, gallery: Tensor) -> Tensor:
    """Cosine distance matrix [num_query, num_gallery] in [0, 2]."""
    query = F.normalize(query, dim=1)
    gallery = F.normalize(gallery, dim=1)
    return 1.0 - query @ gallery.t()


def rank1_success(
    query_feat: Tensor, gallery_feats: Tensor, gallery_pids: Tensor, target_pid: int
) -> bool:
    """True if the closest gallery item to the query has the target identity."""
    distances = pairwise_cosine_distance(query_feat.view(1, -1), gallery_feats).flatten()
    best = torch.argmin(distances)
    return int(gallery_pids[best].item()) == int(target_pid)


def attack_success_rate(
    query_feats: Tensor,
    target_pids: Tensor,
    gallery_feats: Tensor,
    gallery_pids: Tensor,
) -> float:
    """Targeted ASR: fraction of camouflage queries whose Rank-1 match is the
    attacker-specified target identity (paper Sec. IV-A 'Effectiveness')."""
    distances = pairwise_cosine_distance(query_feats, gallery_feats)
    best = distances.argmin(dim=1)
    predicted = gallery_pids[best]
    return float((predicted == target_pids).float().mean().item())


def cmc_and_map(
    query_feats: Tensor,
    query_pids: Tensor,
    gallery_feats: Tensor,
    gallery_pids: Tensor,
    max_rank: int = 10,
) -> tuple[Tensor, float]:
    """Standard ReID Cumulative Matching Characteristic and mean Average Precision.

    Returns (cmc[max_rank], mAP). Assumes every query has at least one
    matching identity in the gallery.
    """
    distances = pairwise_cosine_distance(query_feats, gallery_feats)
    num_q, num_g = distances.shape
    max_rank = min(max_rank, num_g)
    order = distances.argsort(dim=1)
    matches = (gallery_pids[order] == query_pids.view(-1, 1)).float()  # [num_q, num_g]

    cmc = torch.zeros(max_rank, device=distances.device)
    aps = []
    valid = 0
    for i in range(num_q):
        row = matches[i]
        if row.sum() == 0:
            continue
        valid += 1
        cmc_row = row.cumsum(dim=0)
        cmc_row = (cmc_row > 0).float()
        cmc += cmc_row[:max_rank]
        # average precision
        positions = torch.arange(1, num_g + 1, device=distances.device, dtype=torch.float32)
        precision = row.cumsum(dim=0) / positions
        ap = (precision * row).sum() / row.sum()
        aps.append(ap)
    if valid == 0:
        return torch.zeros(max_rank), 0.0
    cmc = cmc / valid
    mean_ap = float(torch.stack(aps).mean().item())
    return cmc.cpu(), mean_ap


def psnr(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    mse = (x - y).square().mean().clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def ssim(x: Tensor, y: Tensor) -> Tensor:
    """Structural similarity in [0, 1] (higher is better)."""
    return 1.0 - ssim_loss(x, y)


@torch.no_grad()
def surrogate_victim_discrepancy(
    surrogate, victim, images: Tensor
) -> dict[str, float]:
    """Empirical surrogate-victim embedding discrepancy delta (paper Eq. 4-7, Sec. IV-E).

    ``surrogate`` may be a SurrogateEnsemble (list output) or a single model.
    Returns the mean and max L2 distance between surrogate and victim
    embeddings over the provided images, which the transferability bound
    predicts should correlate (inversely) with attack success.
    """
    victim_feat = F.normalize(victim(images), dim=1)
    surrogate_out = surrogate(images)
    if isinstance(surrogate_out, (list, tuple)):
        surrogate_feats = [F.normalize(f, dim=1) for f in surrogate_out]
    else:
        surrogate_feats = [F.normalize(surrogate_out, dim=1)]

    # The surrogate and victim embedding spaces need not share dimensionality.
    # We measure discrepancy as the disagreement between their pairwise
    # cosine-distance geometries over the batch, which is dimension-agnostic
    # and still reflects the surrogate-victim gap delta of paper Eq. 4.
    victim_geo = pairwise_cosine_distance(victim_feat, victim_feat)
    diffs = []
    for f in surrogate_feats:
        if f.shape[1] == victim_feat.shape[1]:
            diffs.append((f - victim_feat).norm(dim=1))
        else:
            surrogate_geo = pairwise_cosine_distance(f, f)
            diffs.append((surrogate_geo - victim_geo).abs().mean(dim=1))
    per_image = torch.stack(diffs, dim=0).mean(dim=0)
    return {
        "delta_mean": float(per_image.mean().item()),
        "delta_max": float(per_image.amax().item()),
    }
