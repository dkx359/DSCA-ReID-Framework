import torch

from dsca.losses import DSCALoss, ssim_loss
from dsca.metrics import attack_success_rate, cmc_and_map, psnr, ssim
from dsca.models.surrogates import build_surrogate_ensemble


def test_dsca_loss_returns_finite_components():
    surrogates = build_surrogate_ensemble(embed_dim=64, num_cnn=2, num_transformer=1)
    loss_fn = DSCALoss(surrogates, lambda_gate=0.1)
    generated = torch.rand(2, 3, 64, 32, requires_grad=True)
    source = torch.rand(2, 3, 64, 32)
    target = torch.rand(2, 3, 64, 32)
    outputs = {
        "generated": generated,
        "pred_noise": torch.rand(2, 8, 8, 8),
        "true_noise": torch.rand(2, 8, 8, 8),
        "gates": torch.softmax(torch.rand(2, 3), dim=1),
    }
    losses = loss_fn(outputs, source, target)
    for key in ("total", "id_transfer", "visual", "diffusion", "gate"):
        assert key in losses and torch.isfinite(losses[key])
    losses["total"].backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()


def test_ssim_identity_is_perfect():
    x = torch.rand(1, 3, 32, 32)
    assert ssim_loss(x, x).item() < 1e-3
    assert ssim(x, x).item() > 0.99


def test_psnr_identity_is_high():
    x = torch.rand(1, 3, 32, 32)
    assert psnr(x, x).item() > 60.0


def test_attack_success_rate_perfect_case():
    feats = torch.eye(4)
    pids = torch.tensor([0, 1, 2, 3])
    asr = attack_success_rate(feats, pids, feats, pids)
    assert asr == 1.0


def test_cmc_and_map_perfect_case():
    feats = torch.eye(4)
    pids = torch.tensor([0, 1, 2, 3])
    cmc, mean_ap = cmc_and_map(feats, pids, feats, pids, max_rank=4)
    assert cmc[0].item() == 1.0
    assert abs(mean_ap - 1.0) < 1e-5
