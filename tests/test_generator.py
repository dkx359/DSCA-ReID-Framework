import torch

from dsca.models.diffusion_generator import DSCAGenerator
from dsca.models.extractors import LightweightConditionExtractor


torch.set_num_threads(1)


def _make_generator(condition_dim=32, latent_channels=8, base_channels=4, steps=12):
    return DSCAGenerator(
        pose_channels=17,
        style_dim=64,
        condition_dim=condition_dim,
        latent_channels=latent_channels,
        base_channels=base_channels,
        diffusion_steps=steps,
        beta_start=1e-4,
        beta_end=2e-2,
        mask_scales=[1, 2, 4],
        probability_floor=1e-3,
        lambda_boundary=1.0,
        lambda_skeleton=1.0,
        sharpness=5.0,
    )


def test_generator_forward_runs():
    source = torch.rand(1, 3, 32, 16)
    target = torch.rand(1, 3, 32, 16)
    extractor = LightweightConditionExtractor(pose_channels=17, style_dim=64)
    generator = _make_generator()
    out = generator(source, extractor(target), temperature=1.0)
    assert out["generated"].shape == source.shape
    assert out["pred_noise"].shape == out["true_noise"].shape
    assert out["gates"].shape == (1, 3)


def test_generator_backward_produces_finite_grads():
    source = torch.rand(1, 3, 32, 16)
    target = torch.rand(1, 3, 32, 16)
    extractor = LightweightConditionExtractor(pose_channels=17, style_dim=64)
    generator = _make_generator()
    out = generator(source, extractor(target), temperature=1.0)
    loss = out["generated"].mean() + out["pred_noise"].square().mean()
    loss.backward()
    grads = [p.grad for p in generator.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_generator_ddim_generate_runs():
    source = torch.rand(1, 3, 32, 16)
    target = torch.rand(1, 3, 32, 16)
    extractor = LightweightConditionExtractor(pose_channels=17, style_dim=64)
    generator = _make_generator(steps=12).eval()
    result = generator.generate(source, extractor(target), ddim_steps=2, eta=0.0)
    assert result.shape == source.shape
    assert torch.isfinite(result).all()
    assert (result >= 0).all() and (result <= 1).all()


def test_generator_to_moves_scheduler_buffers():
    generator = _make_generator()
    generator.to("cpu")
    assert generator.scheduler.alpha_bars.device.type == "cpu"
