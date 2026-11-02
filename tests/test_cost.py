import numpy as np

from packbalance.data.cost import FlopModel, cost_stats


def test_cost_model_properties(flop_model: FlopModel) -> None:
    assert flop_model.head_dim == 8
    assert flop_model.d_kv == 16
    assert flop_model.linear_params > 0
    assert flop_model.total_params >= flop_model.linear_params
    assert flop_model.attention_flops_coeff > 0


def test_attention_cost_is_quadratic(flop_model: FlopModel) -> None:
    one_long = flop_model.pack_cost([256])
    two_halves = flop_model.pack_cost([128, 128])
    assert one_long > two_halves
    assert np.isclose(
        one_long - two_halves, flop_model.attention_flops_coeff * (256**2 - 2 * 128**2)
    )


def test_padded_cost_exceeds_packed(flop_model: FlopModel) -> None:
    lengths = [32, 32, 32, 32]
    packed = flop_model.pack_cost(lengths)
    padded = flop_model.padded_batch_cost(lengths, pad_len=64)
    assert padded > packed


def test_cost_stats() -> None:
    stats = cost_stats([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0
    assert stats["cv"] > 0
