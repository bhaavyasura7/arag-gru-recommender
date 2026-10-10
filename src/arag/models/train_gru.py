"""Train the GRU: small grid search, early stopping on validation NDCG@5, 3 seeds.

Saves ``artifacts/gru/{seed}/`` = model.pt + item_map.json + config.json.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from arag import util
from arag.config import get_config
from arag.eval.metrics import ndcg_at_k, rank_of
from arag.models.gru import GRU4Rec, save_model

PAD = 0
OOV = 1


def build_vocab(train_item_ids) -> tuple[dict[str, int], dict[int, str]]:
    ids = sorted(set(train_item_ids))
    item2idx = {iid: idx for idx, iid in enumerate(ids, start=2)}
    idx2item = {idx: iid for iid, idx in item2idx.items()}
    return item2idx, idx2item


def build_train_array(sequences, item2idx, seq_len, max_users, logger) -> np.ndarray:
    rows = []
    for user_id, row in sequences.iterrows():
        idxs = [item2idx[i] for i in row["train_items"] if i in item2idx][-seq_len:]
        if len(idxs) >= 2:
            rows.append(idxs)
        if max_users and len(rows) >= max_users:
            break
    if max_users and len(sequences) > max_users:
        logger.info("training users capped to %d (of %d)", max_users, len(sequences))
    n = len(rows)
    arr = np.zeros((n, seq_len), dtype=np.int32)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r  # right-pad (pack_padded_sequence expects valid items first)
    return arr


def build_val_data(sequences, valid_lists, item2idx) -> list[dict]:
    hist_by_user = {row["user_id"]: row["train_items"] for _, row in sequences.iterrows()}
    val_data = []
    for _, row in valid_lists.iterrows():
        uid = row["user_id"]
        if uid not in hist_by_user:
            continue
        history_idx = [item2idx[i] for i in hist_by_user[uid] if i in item2idx]
        candidate_idx = [item2idx.get(i, OOV) for i in row["candidates"]]
        target_idx = item2idx.get(row["target"], OOV)
        val_data.append(
            {
                "history": history_idx,
                "candidates": candidate_idx,
                "target": target_idx,
                "candidate_ids": row["candidates"],
                "target_id": row["target"],
            }
        )
    return val_data


def train_model(
    model: GRU4Rec,
    train_arr: np.ndarray,
    val_data: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    logger,
) -> tuple[GRU4Rec, list[float]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_arr.shape[0]
    seq_len = train_arr.shape[1]
    idx = np.arange(n)

    best_ndcg = -1.0
    best_state = None
    wait = 0
    ndcg_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(idx)
        order = train_arr[idx]
        order = order[np.argsort(-(order != PAD).sum(axis=1), kind="stable")]  # longest first
        total_loss = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            x_in = torch.tensor(batch[:, :-1], dtype=torch.long)
            y = torch.tensor(batch[:, 1:], dtype=torch.long)
            lengths = (x_in != PAD).sum(dim=1)
            mask = x_in != PAD
            logits = model(x_in, lengths)  # (B, T-1, vocab)
            mask = x_in != PAD
            logits_flat = logits.reshape(-1, logits.shape[-1])
            y_flat = y.reshape(-1)
            mask_flat = mask.reshape(-1)
            loss = F.cross_entropy(logits_flat[mask_flat], y_flat[mask_flat], ignore_index=PAD)
            n_valid = int(mask.sum().item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * n_valid
            nb += n_valid

        val_ndcg = evaluate(model, val_data)
        ndcg_history.append(val_ndcg)
        logger.info(
            "epoch %d: loss %.4f  val NDCG@5 %.4f",
            epoch,
            total_loss / max(nb, 1),
            val_ndcg,
        )
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, ndcg_history


def evaluate(model: GRU4Rec, val_data: list[dict]) -> float:
    model.eval()
    ndcgs = []
    with torch.no_grad():
        for v in val_data:
            if not v["history"]:
                continue
            x = torch.tensor([v["history"][-50:]], dtype=torch.long)
            h_u = model.encode(x)[0]
            cidx = torch.tensor(v["candidates"], dtype=torch.long)
            scores = h_u @ model.embedding(cidx).T
            order = torch.argsort(scores, descending=True).tolist()
            ranking = [v["candidate_ids"][i] for i in order]
            rank = rank_of(v["target_id"], ranking)
            ndcgs.append(ndcg_at_k(rank, 5))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def main() -> None:
    logger = util.setup_logging("arag.models.train_gru")
    cfg = get_config()
    proc_dir: Path = cfg.path("data_processed")
    lists_dir: Path = cfg.path("lists")
    gru_dir: Path = cfg.path("gru")

    interactions = pd.read_parquet(proc_dir / "interactions.parquet")
    sequences = pd.read_parquet(lists_dir / "sequences.parquet")
    valid_lists = pd.read_parquet(lists_dir / "valid_lists.parquet")

    train_items = interactions.loc[interactions["split"] == "train", "item_id"].astype(str)
    item2idx, _ = build_vocab(train_items)
    logger.info("train vocab: %d items", len(item2idx))

    seq_len = cfg.get("gru.seq_len", 50)
    emb_dim = cfg.get("gru.embedding_dim", 64)
    train_arr = build_train_array(
        sequences, item2idx, seq_len, cfg.get("gru.max_train_users", 0), logger
    )
    logger.info("training sequences: %d", train_arr.shape[0])
    val_data = build_val_data(sequences, valid_lists, item2idx)
    logger.info("validation users for early stop: %d", len(val_data))

    epochs = cfg.get("gru.max_epochs", 30)
    batch_size = cfg.get("gru.batch_size", 256)
    lr = cfg.get("gru.lr", 0.001)
    patience = cfg.get("gru.patience", 3)

    # grid search on validation (seed 0)
    best = None
    for hidden in cfg.get("gru.hidden_grid", [64, 128]):
        for dropout in cfg.get("gru.dropout_grid", [0.2, 0.5]):
            torch.manual_seed(0)
            model = GRU4Rec(len(item2idx) + 2, emb_dim, hidden, dropout)
            t0 = time.time()
            _, history = train_model(model, train_arr, val_data, epochs, batch_size, lr, patience, logger)
            score = max(history)
            logger.info("grid hidden=%d dropout=%.1f -> val NDCG@5 %.4f (%.0fs)", hidden, dropout, score, time.time() - t0)
            if best is None or score > best[0]:
                best = (score, hidden, dropout)

    best_ndcg, hidden, dropout = best
    logger.info("best config: hidden=%d dropout=%.1f NDCG@5=%.4f", hidden, dropout, best_ndcg)

    for seed in cfg.get("gru.seeds", [0, 1, 2]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = GRU4Rec(len(item2idx) + 2, emb_dim, hidden, dropout)
        t0 = time.time()
        model, history = train_model(model, train_arr, val_data, epochs, batch_size, lr, patience, logger)
        out_dir = gru_dir / str(seed)
        save_model(
            out_dir,
            model,
            item2idx,
            {
                "emb_dim": emb_dim,
                "hidden": hidden,
                "dropout": dropout,
                "seq_len": seq_len,
                "seed": seed,
                "best_val_ndcg5": max(history),
                "vocab_size": len(item2idx),
            },
        )
        logger.info("saved seed=%d -> %s (%.0fs, val NDCG@5=%.4f)", seed, out_dir, time.time() - t0, max(history))


if __name__ == "__main__":
    main()
