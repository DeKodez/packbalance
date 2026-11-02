import torch
import torch.nn.functional as F

from packbalance.config import DataConfig, ModelConfig
from packbalance.data.cost import FlopModel
from packbalance.data.dataset import PackedDataset
from packbalance.data.packer import build_packs
from packbalance.data.store import TokenStore
from packbalance.model.attention import eager_attention
from packbalance.model.llama import LlamaModel


def test_eager_attention_matches_dense_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
    mask = (doc_ids == doc_ids.T) & torch.tril(torch.ones(8, 8, dtype=torch.bool))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
    out = eager_attention(q, k, v, doc_ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_packed_model_matches_isolated_documents(
    store: TokenStore, data_cfg: DataConfig, model_cfg: ModelConfig, flop_model: FlopModel
) -> None:
    torch.manual_seed(0)
    packs = build_packs(store, data_cfg, flop_model, split="train")
    pack = next(p for p in packs if p.kind == "packed" and len(p.segments) >= 2)
    ds = PackedDataset(store, [pack], data_cfg)
    item = ds[0]
    model = LlamaModel(model_cfg).eval()
    with torch.no_grad():
        packed = model(
            item["input_ids"].unsqueeze(0),
            item["position_ids"].unsqueeze(0),
            item["doc_ids"].unsqueeze(0),
        )
        offset = 0
        for seg in pack.segments:
            tokens = torch.from_numpy(store.read(seg.start, seg.length))
            isolated = model(
                tokens.unsqueeze(0),
                torch.arange(seg.length).unsqueeze(0),
                torch.zeros(1, seg.length, dtype=torch.long),
            )
            torch.testing.assert_close(
                packed[0, offset : offset + seg.length],
                isolated[0],
                atol=1e-5,
                rtol=1e-5,
            )
            offset += seg.length


def test_flex_attention_matches_eager_on_cuda() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is required for FlexAttention")
    from packbalance.model.attention import build_doc_block_mask, flex_attention

    torch.manual_seed(0)
    q = torch.randn(1, 2, 64, 16, device="cuda")
    k = torch.randn(1, 2, 64, 16, device="cuda")
    v = torch.randn(1, 2, 64, 16, device="cuda")
    doc_ids = torch.tensor([[0] * 20 + [1] * 44], device="cuda")
    mask = build_doc_block_mask(doc_ids, n_heads=2)
    out = flex_attention(q, k, v, block_mask=mask)
    ref = eager_attention(q, k, v, doc_ids)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
