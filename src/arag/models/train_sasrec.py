"""Train SASRec with the GRU's exact protocol, then rank the frozen test lists for all users.

Same vocabulary, training sequences, validation users, early stopping (validation NDCG@5,
patience) and seeds as ``train_gru``. Small grid over dropout and blocks on seed 0.

Saves ``artifacts/sasrec/{seed}/`` and ``artifacts/predictions/sasrec_seed{seed}_test.parquet``
(same format as the other systems' predictions).

    python -m arag.models.train_sasrec
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from arag import util
from arag.config import get_config
from arag.eval.metrics import ndcg_at_k, rank_of
from arag.models.sasrec import SASRec, save_model
from arag.models.train_gru import OOV, PAD, build_train_array, build_val_data, build_vocab


def _batch_tensor(hists: list[list[int]]) -> torch.Tensor:
    x = torch.zeros(len(hists), max(len(h) for h in hists), dtype=torch.long)
    for i, h in enumerate(hists):
        x[i, : len(h)] = torch.tensor(h, dtype=torch.long)
    return x


def rank_lists(model: SASRec, rows: list[dict], seq_len: int, chunk: int = 4096) -> list[list[str]]:
    """Rank each row's candidates by h_u . e_i (ties broken by item id, as GRUScorer)."""
    model.eval()
    out: list[list[str]] = []
    with torch.no_grad():
        for c0 in range(0, len(rows), chunk):
            part = rows[c0 : c0 + chunk]
            hists = [(r["history"][-seq_len:] or [OOV]) for r in part]
            h_u = model.encode(_batch_tensor(hists))
            cidx = torch.tensor([r["candidates"] for r in part], dtype=torch.long)
            scores = (model.item_emb(cidx) * h_u[:, None, :]).sum(-1).tolist()
            for r, sc in zip(part, scores):
                pairs = sorted(zip(r["candidate_ids"], sc), key=lambda p: (-p[1], p[0]))
                out.append([iid for iid, _ in pairs])
    return out


def evaluate(model: SASRec, val_data: list[dict], seq_len: int) -> float:
    items = [v for v in val_data if v["history"]]
    rankings = rank_lists(model, items, seq_len)
    return float(np.mean([ndcg_at_k(rank_of(v["target_id"], r), 5) for v, r in zip(items, rankings)]))


def train_model(model, train_arr, val_data, epochs, batch_size, lr, patience, logger):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, seq_len = train_arr.shape
    idx = np.arange(n)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(idx)
        order = train_arr[idx]
        order = order[np.argsort(-(order != PAD).sum(axis=1), kind="stable")]  # longest first
        total, nb = 0.0, 0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            t = int((batch != PAD).sum(axis=1).max())
            x_in = torch.tensor(batch[:, : t - 1], dtype=torch.long)
            y = torch.tensor(batch[:, 1:t], dtype=torch.long)
            mask = (x_in != PAD) & (y != PAD)
            out = model.step_outputs(x_in)[mask]
            loss = F.cross_entropy(F.linear(out, model.all_item_emb()), y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            k = int(mask.sum())
            total += float(loss.item()) * k
            nb += k
        val = evaluate(model, val_data, seq_len)
        history.append(val)
        logger.info("epoch %d: loss %.4f  val NDCG@5 %.4f", epoch, total / max(nb, 1), val)
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                logger.info("early stop at epoch %d", epoch)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def test_rows(cfg, interactions: pd.DataFrame, item2idx: dict[str, int]) -> list[dict]:
    """Test-time history = every interaction strictly before the test target (as ``history_before``)."""
    seqs = pd.read_parquet(cfg.path("lists") / "sequences.parquet").set_index("user_id")
    lists = pd.read_parquet(cfg.path("lists") / "test_lists.parquet")
    inter = interactions.merge(seqs[["test_ts", "test_item"]], left_on="user_id", right_index=True)
    inter = inter[(inter["timestamp"] < inter["test_ts"]) & (inter["item_id"] != inter["test_item"])]
    inter = inter.sort_values(["user_id", "timestamp", "item_id"])
    hist = inter.groupby("user_id")["item_id"].apply(list).to_dict()
    rows = []
    for _, r in lists.iterrows():
        cands = [str(c) for c in r["candidates"]]
        rows.append({
            "user_id": r["user_id"],
            "history": [item2idx.get(i, OOV) for i in hist.get(r["user_id"], [])],
            "candidates": [item2idx.get(c, OOV) for c in cands],
            "candidate_ids": cands,
            "target_id": str(r["target"]),
        })
    return rows


def main() -> None:
    logger = util.setup_logging("arag.models.train_sasrec")
    cfg = get_config()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    interactions = pd.read_parquet(cfg.path("data_processed") / "interactions.parquet")
    interactions["item_id"] = interactions["item_id"].astype(str)
    sequences = pd.read_parquet(cfg.path("lists") / "sequences.parquet")
    valid_lists = pd.read_parquet(cfg.path("lists") / "valid_lists.parquet")

    item2idx, _ = build_vocab(interactions.loc[interactions["split"] == "train", "item_id"])
    seq_len = cfg.get("gru.seq_len", 50)
    emb_dim = cfg.get("gru.embedding_dim", 64)
    train_arr = build_train_array(sequences, item2idx, seq_len, cfg.get("gru.max_train_users", 0), logger)
    val_data = build_val_data(sequences, valid_lists, item2idx)
    logger.info("vocab %d items, %d training sequences, %d validation users", len(item2idx), len(train_arr), len(val_data))

    epochs = cfg.get("sasrec.max_epochs", 30)
    batch_size = cfg.get("gru.batch_size", 256)
    lr = cfg.get("gru.lr", 0.001)
    patience = cfg.get("gru.patience", 3)
    n_heads = cfg.get("sasrec.n_heads", 2)

    def new_model(n_blocks, dropout):
        return SASRec(len(item2idx) + 2, emb_dim, n_blocks, n_heads, dropout, seq_len)

    best = None
    for n_blocks in cfg.get("sasrec.blocks_grid", [2]):
        for dropout in cfg.get("sasrec.dropout_grid", [0.2, 0.5]):
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            t0 = time.time()
            _, hist = train_model(new_model(n_blocks, dropout), train_arr, val_data, epochs, batch_size, lr, patience, logger)
            logger.info("grid blocks=%d dropout=%.1f -> val NDCG@5 %.4f (%.0fs)", n_blocks, dropout, max(hist), time.time() - t0)
            if best is None or max(hist) > best[0]:
                best = (max(hist), n_blocks, dropout)
    _, n_blocks, dropout = best
    logger.info("best config: blocks=%d dropout=%.1f", n_blocks, dropout)

    rows = test_rows(cfg, interactions, item2idx)
    logger.info("test users: %d", len(rows))
    pred_dir = cfg.path("artifacts") / "predictions"
    for seed in cfg.get("gru.seeds", [0, 1, 2]):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        t0 = time.time()
        model, hist = train_model(new_model(n_blocks, dropout), train_arr, val_data, epochs, batch_size, lr, patience, logger)
        save_model(cfg.path("artifacts") / "sasrec" / str(seed), model, item2idx, {
            "emb_dim": emb_dim, "n_blocks": n_blocks, "n_heads": n_heads, "dropout": dropout,
            "seq_len": seq_len, "seed": seed, "best_val_ndcg5": max(hist), "vocab_size": len(item2idx),
        })
        rankings = rank_lists(model, rows, seq_len)
        pd.DataFrame({
            "user_id": [r["user_id"] for r in rows],
            "ranked_ids": rankings,
            "target": [r["target_id"] for r in rows],
            "rank": [rank_of(r["target_id"], rk) for r, rk in zip(rows, rankings)],
        }).to_parquet(pred_dir / f"sasrec_seed{seed}_test.parquet", index=False)
        ndcg = np.mean([ndcg_at_k(rank_of(r["target_id"], rk), 5) for r, rk in zip(rows, rankings)])
        logger.info("seed %d: val NDCG@5 %.4f, test NDCG@5 (all users) %.4f (%.0fs)", seed, max(hist), ndcg, time.time() - t0)


if __name__ == "__main__":
    main()
