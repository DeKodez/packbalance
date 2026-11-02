import torch

from packbalance.config import ModelConfig
from packbalance.model.llama import LlamaForCausalLM, count_parameters


def _small_model(cfg: ModelConfig) -> LlamaForCausalLM:
    return LlamaForCausalLM(cfg)


def test_forward_backward_and_loss(model_cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    model = _small_model(model_cfg)
    assert count_parameters(model) > 0
    input_ids = torch.randint(0, model_cfg.vocab_size, (1, 16))
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    position_ids = torch.arange(16).unsqueeze(0)
    doc_ids = torch.zeros_like(input_ids)
    out = model(input_ids, labels=labels, position_ids=position_ids, doc_ids=doc_ids)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.hidden_states is not None
    assert out.hidden_states.shape == (1, 16, model_cfg.d_model)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_overfit_one_tiny_batch(model_cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    model = _small_model(model_cfg)
    input_ids = torch.randint(0, model_cfg.vocab_size, (1, 32))
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    position_ids = torch.arange(32).unsqueeze(0)
    doc_ids = torch.zeros_like(input_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, labels=labels, position_ids=position_ids, doc_ids=doc_ids)
        assert out.loss is not None
        loss = out.loss / int((labels != -100).sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.85, (losses[0], losses[-1])


def test_generate_shape(model_cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    model = _small_model(model_cfg)
    input_ids = torch.randint(0, model_cfg.vocab_size, (1, 4))
    out = model.generate(input_ids, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 8)
