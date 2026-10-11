"""Train the GRU: small grid search, early stopping on validation NDCG@5, 3 seeds.

Saves ``artifacts/gru/{seed}/`` = model.pt + item_map.json + config.json.

``--text``: text-aware variant (item vector = ID embedding + projected MiniLM text embedding).
Grid over ``gru_text.hidden_grid`` x ``gru_text.dropout_grid``, saved to ``artifacts/gru_text/{seed}/``.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from arag import util
from arag.config import get_config
from arag.eval.metrics import ndcg_at_k, rank_of
from arag.models.gru import GRU4Rec, load_text_feats, save_model

PAD = 0
OOV = 1
MAX_LOGIT_ELEMS = 250_000_000  # ~1 GB of float32 logits per part (larger parts thrash 16 GB unified memory)


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
    dev = next(model.parameters()).device
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
            t = int((batch != PAD).sum(axis=1).max())  # trim padding columns beyond the longest row
            x_in = torch.tensor(batch[:, : t - 1], dtype=torch.long, device=dev)
            y = torch.tensor(batch[:, 1:t], dtype=torch.long, device=dev)
            lengths = (x_in != PAD).sum(dim=1)
            # a position is trained iff it has a real next item (same as masking full logits,
            # but logits are only materialised for those positions: far less memory)
            mask = (x_in != PAD) & (y != PAD)
            n_valid = int(mask.sum().item())
            opt.zero_grad()
            # Split the batch by users when its logit matrix (n_valid x vocab) would exceed
            # MAX_LOGIT_ELEMS (a GPU buffer limit on large vocabularies). Each part contributes
            # sum-CE / n_valid, so the accumulated gradient equals the full-batch mean-CE gradient.
            vocab = model.vocab_size
            per_row = mask.sum(dim=1).tolist()
            parts, cur, acc = [], [], 0
            for r, k in enumerate(per_row):
                if cur and (acc + k) * vocab > MAX_LOGIT_ELEMS:
                    parts.append(cur)
                    cur, acc = [], 0
                cur.append(r)
                acc += k
            parts.append(cur)
            batch_loss = 0.0
            for rows in parts:
                if len(parts) == 1:
                    xs, ys, ls, ms = x_in, y, lengths, mask
                else:
                    sel = torch.tensor(rows, device=dev)
                    xs, ys, ls, ms = x_in[sel], y[sel], lengths[sel], mask[sel]
                out = model.step_outputs(xs, ls)[ms]  # (n_part, emb)
                logits = F.linear(out, model.all_item_emb())  # (n_part, vocab)
                loss = F.cross_entropy(logits, ys[ms], reduction="sum") / n_valid
                loss.backward()
                batch_loss += float(loss.item())
            opt.step()
            total_loss += batch_loss * n_valid
            nb += n_valid

        val_ndcg = evaluate(model, val_data, seq_len=seq_len)
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


def evaluate(model: GRU4Rec, val_data: list[dict], seq_len: int = 50, chunk: int = 4096) -> float:
    """Validation NDCG@5 on the frozen lists. History capped to the last ``seq_len`` items
    (as ``GRUScorer`` does at test time); users encoded in chunks to bound memory."""
    model.eval()
    dev = next(model.parameters()).device
    items = [v for v in val_data if v["history"]]
    if not items:
        return 0.0
    ndcgs = []
    with torch.no_grad():
        for c0 in range(0, len(items), chunk):
            part = items[c0 : c0 + chunk]
            hists = [v["history"][-seq_len:] for v in part]
            x = torch.zeros(len(part), max(len(h) for h in hists), dtype=torch.long)
            for i, h in enumerate(hists):
                x[i, : len(h)] = torch.tensor(h, dtype=torch.long)
            h_u = model.encode(x.to(dev))  # (B, emb)
            cidx = torch.tensor([v["candidates"] for v in part], dtype=torch.long, device=dev)  # (B, 20)
            scores = (model.item_emb(cidx) * h_u[:, None, :]).sum(-1)  # (B, 20)
            orders = torch.argsort(scores, dim=1, descending=True).tolist()
            for v, order in zip(part, orders):
                ranking = [v["candidate_ids"][j] for j in order]
                ndcgs.append(ndcg_at_k(rank_of(v["target_id"], ranking), 5))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def main() -> None:
    text = "--text" in sys.argv[1:]
    logger = util.setup_logging("arag.models.train_gru")
    cfg = get_config()
    proc_dir: Path = cfg.path("data_processed")
    lists_dir: Path = cfg.path("lists")
    gru_dir: Path = cfg.path("gru_text" if text else "gru")
    section = "gru_text" if text else "gru"

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

    text_feats = None
    text_feats_dir = ""
    if text:
        text_feats_dir = str(cfg.path("faiss"))
        text_feats = load_text_feats(text_feats_dir, item2idx)
        n_text = int((text_feats.abs().sum(dim=1) > 0).sum())
        logger.info("text features: %d of %d vocab items", n_text, len(item2idx))

    dev_name = cfg.get("gru.device", "cpu")
    if dev_name == "auto":
        dev_name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    logger.info("training device: %s", device)

    def new_model(hidden, dropout):
        return GRU4Rec(len(item2idx) + 2, emb_dim, hidden, dropout, text_feats=text_feats).to(device)

    epochs = cfg.get("gru.max_epochs", 30)
    batch_size = cfg.get("gru.batch_size", 256)
    lr = cfg.get("gru.lr", 0.001)
    patience = cfg.get("gru.patience", 3)

    # grid search on validation (seed 0)
    best = None
    hidden_grid = cfg.get(f"{section}.hidden_grid", [64, 128])
    dropout_grid = cfg.get(f"{section}.dropout_grid", [0.2, 0.5])
    for hidden in hidden_grid:
        for dropout in dropout_grid:
            if len(hidden_grid) * len(dropout_grid) == 1:
                best = (None, hidden, dropout)  # single config: no grid run needed
                break
            torch.manual_seed(0)
            model = new_model(hidden, dropout)
            t0 = time.time()
            _, history = train_model(model, train_arr, val_data, epochs, batch_size, lr, patience, logger)
            score = max(history)
            logger.info("grid hidden=%d dropout=%.1f -> val NDCG@5 %.4f (%.0fs)", hidden, dropout, score, time.time() - t0)
            if best is None or score > best[0]:
                best = (score, hidden, dropout)

    best_ndcg, hidden, dropout = best
    logger.info("best config: hidden=%d dropout=%.1f NDCG@5=%s", hidden, dropout, best_ndcg)

    for seed in cfg.get("gru.seeds", [0, 1, 2]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = new_model(hidden, dropout)
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
                "text_feats_dir": text_feats_dir,
            },
        )
        logger.info("saved seed=%d -> %s (%.0fs, val NDCG@5=%.4f)", seed, out_dir, time.time() - t0, max(history))


if __name__ == "__main__":
    main()
