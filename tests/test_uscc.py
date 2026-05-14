import torch

from dsca.models.extractors import LightweightConditionExtractor
from dsca.models.uscc import UnifiedSemanticConditionComposer


def _make_uscc(condition_dim=128):
    return UnifiedSemanticConditionComposer(
        pose_channels=17,
        style_dim=64,
        condition_dim=condition_dim,
        mask_scales=[1, 2, 4],
    )


def test_uscc_outputs_have_expected_shapes():
    x = torch.rand(2, 3, 64, 32)
    extractor = LightweightConditionExtractor(pose_channels=17, style_dim=64)
    conditions = extractor(x)
    uscc = _make_uscc(128)
    out = uscc(conditions, temperature=1.0)
    assert out.condition.shape == (2, 128)
    assert out.gates.shape == (2, 3)
    assert torch.allclose(out.gates.sum(dim=1), torch.ones(2), atol=1e-5)
    assert "2" in out.edge_masks
    assert out.skeleton.shape == (2, 1, 64, 32)


def test_uscc_gates_respect_probability_floor():
    x = torch.rand(2, 3, 64, 32)
    conditions = LightweightConditionExtractor()(x)
    uscc = _make_uscc(64)
    out = uscc(conditions, temperature=0.5)
    assert (out.gates >= 0).all()
    # gate entropy is a finite scalar usable as a regulariser
    ent = UnifiedSemanticConditionComposer.gate_entropy(out.gates)
    assert ent.dim() == 0 and torch.isfinite(ent)


def test_uscc_temperature_softmax_is_stable_for_large_logits():
    uscc = _make_uscc(32)
    logits = torch.tensor([[1e3, -1e3, 0.0], [0.0, 0.0, 0.0]])
    gates = uscc._temperature_softmax(logits, temperature=1.0)
    assert torch.isfinite(gates).all()
    assert torch.allclose(gates.sum(dim=1), torch.ones(2), atol=1e-5)
