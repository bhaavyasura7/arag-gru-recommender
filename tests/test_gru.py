import logging

import numpy as np
import torch

from arag.models.gru import GRU4Rec, load_model, save_model
from arag.models.train_gru import evaluate, train_model


def _quiet_logger():
    lg = logging.getLogger("test_gru")
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
    return lg


def test_padding_masked_same_hidden_state():
    torch.manual_seed(0)
    m = GRU4Rec(vocab_size=20, emb_dim=8, hidden=16)
    m.eval()
    seq = torch.tensor([[2, 3, 4]], dtype=torch.long)
    padded = torch.tensor([[2, 3, 4, 0, 0]], dtype=torch.long)  # right-padded
    with torch.no_grad():
        h1 = m.encode(seq)
        h2 = m.encode(padded)
    assert torch.allclose(h1, h2, atol=1e-6)


def test_save_load_reproduces_scores(tmp_path):
    torch.manual_seed(1)
    m = GRU4Rec(vocab_size=20, emb_dim=8, hidden=16)
    item2idx = {f"i{k}": k for k in range(2, 20)}  # indices 2..19 -> vocab 20
    save_model(tmp_path / "m", m, item2idx, {"emb_dim": 8, "hidden": 16, "dropout": 0.2, "seq_len": 50})

    m2, item2idx2, _ = load_model(tmp_path / "m")
    assert item2idx2 == item2idx
    x = torch.tensor([[2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(m.encode(x), m2.encode(x), atol=1e-6)


def test_gru_beats_random_on_toy_sequence():
    items = [f"i{k}" for k in range(2, 12)]
    # next-item rule: i_k -> i_{k+1} (cycle)
    seqs = [[(u + t) % 10 + 2 for t in range(15)] for u in range(200)]
    train_arr = np.zeros((200, 50), dtype=np.int32)
    for i, s in enumerate(seqs):
        train_arr[i, : len(s)] = s

    val = []
    for k in range(10):
        idx = k + 2
        succ = (k + 1) % 10 + 2
        val.append(
            {
                "history": [idx],
                "candidates": list(range(2, 12)),
                "target": succ,
                "candidate_ids": items,
                "target_id": items[(k + 1) % 10],
            }
        )

    torch.manual_seed(0)
    model = GRU4Rec(vocab_size=12, emb_dim=8, hidden=16, dropout=0.0)
    model, history = train_model(
        model, train_arr, val, epochs=8, batch_size=64, lr=0.01, patience=5, logger=_quiet_logger()
    )
    ndcg = evaluate(model, val)
    # perfect successor prediction -> rank 1 -> NDCG@5 = 1.0; random ~0.25
    assert ndcg > 0.5, f"expected GRU to learn the toy sequence, got NDCG@5={ndcg}"
