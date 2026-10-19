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
        model, train_arr, val, epochs=30, batch_size=64, lr=0.01, patience=30, logger=_quiet_logger()
    )
    ndcg = evaluate(model, val)
    # perfect successor prediction -> rank 1 -> NDCG@5 = 1.0; random ~0.25
    assert ndcg > 0.5, f"expected GRU to learn the toy sequence, got NDCG@5={ndcg}"


def _text_dir(tmp_path, item2idx, dim=6):
    d = tmp_path / "faiss"
    d.mkdir()
    ids = np.array(list(item2idx), dtype=object)
    rng = np.random.default_rng(0)
    np.save(d / "item_ids.npy", ids, allow_pickle=True)
    np.save(d / "item_embeddings.npy", rng.normal(size=(len(ids), dim)).astype(np.float32))
    return d


def test_text_gru_padding_and_save_load(tmp_path):
    from arag.models.gru import load_text_feats

    item2idx = {f"i{k}": k for k in range(2, 20)}
    d = _text_dir(tmp_path, item2idx)
    feats = load_text_feats(d, item2idx)
    assert feats.shape == (20, 6)
    assert torch.all(feats[0] == 0) and torch.all(feats[1] == 0)  # pad / OOV

    torch.manual_seed(2)
    m = GRU4Rec(vocab_size=20, emb_dim=8, hidden=16, text_feats=feats)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m.encode(torch.tensor([[2, 3, 4]])), m.encode(torch.tensor([[2, 3, 4, 0]])), atol=1e-6)
    assert "text_feats" not in m.state_dict()  # frozen features are reloaded, not saved

    save_model(tmp_path / "m", m, item2idx, {"emb_dim": 8, "hidden": 16, "dropout": 0.2, "text_feats_dir": str(d)})
    m2, _, _ = load_model(tmp_path / "m")
    x = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(m.encode(x), m2.encode(x), atol=1e-6)
        assert torch.allclose(m.all_item_emb()[5], m2.item_emb(torch.tensor(5)), atol=1e-6)
