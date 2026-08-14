"""Correctness tests for the transformer-timescale architecture hierarchy."""

import pytest
import torch

from timescales.transformers.sequence_models import create_sequence_model


BASE_KWARGS = {
    "input_size": 3,
    "output_size": 3,
    "d_model": 12,
    "n_heads": 3,
    "max_context_length": 16,
    "d_mlp": 24,
    "residual_init_scale": 0.08,
    "residual_gain": 1.0,
    "attention_logit_scale": 1.0,
    "output_coupling_gamma": 0.1,
}


@pytest.mark.parametrize(
    "architecture",
    [
        "linear_fir",
        "static_linear_attention",
        "linear_attention",
        "softmax_attention",
        "softmax_tanh",
    ],
)
def test_all_architectures_produce_component_predictions(architecture):
    torch.manual_seed(0)
    model = create_sequence_model(architecture, **BASE_KWARGS)
    inputs = torch.randn(5, 16, 3)
    outputs, cache = model(inputs, return_cache=True)
    assert outputs.shape == (5, 16, 3)
    assert "residual_streams" in cache
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("architecture", ["linear_fir", "static_linear_attention"])
def test_claimed_linear_models_obey_superposition(architecture):
    torch.manual_seed(1)
    model = create_sequence_model(architecture, **BASE_KWARGS)
    first = torch.randn(2, 16, 3)
    second = torch.randn(2, 16, 3)
    alpha, beta = 0.7, -1.3
    combined = model(alpha * first + beta * second)
    expected = alpha * model(first) + beta * model(second)
    torch.testing.assert_close(combined, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "architecture",
    [
        "linear_fir",
        "static_linear_attention",
        "linear_attention",
        "softmax_attention",
        "softmax_tanh",
    ],
)
def test_all_models_are_causal(architecture):
    torch.manual_seed(2)
    model = create_sequence_model(architecture, **BASE_KWARGS)
    original = torch.randn(2, 16, 3)
    perturbed = original.clone()
    perturbed[:, 10:, :] += 100.0 * torch.randn_like(perturbed[:, 10:, :])
    before = model(original)
    after = model(perturbed)
    torch.testing.assert_close(before[:, :10], after[:, :10], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("architecture", ["softmax_attention", "softmax_tanh"])
def test_softmax_attention_is_normalized_and_masked(architecture):
    torch.manual_seed(3)
    model = create_sequence_model(architecture, **BASE_KWARGS)
    _, cache = model(torch.randn(2, 16, 3), return_cache=True)
    attention = cache["attention_patterns"][0]
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))
    )
    future_mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    assert torch.equal(
        attention.masked_select(future_mask[None, None]),
        torch.zeros_like(attention.masked_select(future_mask[None, None])),
    )


def test_output_gamma_changes_scale_without_changing_initial_draw():
    torch.manual_seed(5)
    first = create_sequence_model(
        "static_linear_attention", **{**BASE_KWARGS, "output_coupling_gamma": 1.0}
    )
    torch.manual_seed(5)
    second = create_sequence_model(
        "static_linear_attention", **{**BASE_KWARGS, "output_coupling_gamma": 0.5}
    )
    inputs = torch.randn(2, 16, 3)
    torch.testing.assert_close(second(inputs), 2.0 * first(inputs))


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown sequence architecture"):
        create_sequence_model("not_a_model", **BASE_KWARGS)
