"""
SASRec-style Transformer recommender for Sim4Rec
Ashita Singh
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import pyspark.sql.functions as sf
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None  

# ═════════════════════ helper utilities ════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_pandas(df):
    if df is None:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError(f"Expected pandas or PySpark DataFrame, got {type(df)}")

# ═════════════════════ dataset builder ═════════════════════════════
class _SeqDataset(Dataset):
    """Generates (sequence, positive, negative, price) samples."""

    def __init__(self, seqs: Dict[int, List[int]], prices: Dict[int, float],
                 all_items: List[int], max_len: int):
        super().__init__()
        self.max_len = max_len
        self.all_items = np.array(all_items)

        self.seqs, self.pos, self.neg, self.price = [], [], [], []
        for items in seqs.values():
            pool = set(items)
            for t in range(1, len(items)):
                s = items[max(0, t - max_len): t]
                p = items[t]
                n = np.random.choice(self.all_items)
                while int(n) in pool:
                    n = np.random.choice(self.all_items)

                self.seqs.append(s)
                self.pos.append(p)
                self.neg.append(int(n))
                self.price.append(prices.get(p, 1.0))

    def __len__(self):  return len(self.seqs)

    def __getitem__(self, idx):
        seq = [0]*(self.max_len - len(self.seqs[idx])) + self.seqs[idx]
        return (torch.tensor(seq, dtype=torch.long),
                torch.tensor(self.pos[idx], dtype=torch.long),
                torch.tensor(self.neg[idx], dtype=torch.long),
                torch.tensor(self.price[idx], dtype=torch.float))

# ═══════════════════ transformer encoder ═══════════════════════════
class _Block(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=p)
        self.ffn  = nn.Sequential(
            nn.Linear(d, 4*d), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(4*d, d), nn.Dropout(p))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + a)
        f = self.ffn(x)
        return self.n2(x + f)


class _SASRec(nn.Module):
    def __init__(self, n_items: int, d: int, layers: int,
                 heads: int, p: float, max_len: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, d)
        self.blocks   = nn.ModuleList([_Block(d, heads, p) for _ in range(layers)])
        self.dp = nn.Dropout(p)
        self.max_len = max_len

    def forward(self, seq):                                   # seq [B,L]
        pos = torch.arange(seq.size(1), device=seq.device)[None]
        x   = self.item_emb(seq) + self.pos_emb(pos)
        x   = self.dp(x).transpose(0,1)                       # [L,B,D]
        mask = torch.triu(torch.ones(seq.size(1), seq.size(1),
                                     device=seq.device), 1).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        return x.transpose(0,1)                               # [B,L,D]

# ═══════════════════ public recommender ════════════════════════════
class TransformerRecommender:
    """Revenue-aware SASRec recommender with price-aware inference."""

    HP = dict(dim=128, max_len=50, layers=3, heads=4, dropout=0.2,
              batch=512, lr=5e-4, wd=1e-4, epochs=15, patience=3,
              price_exp=1.0)

    def __init__(self, seed: int = 42, device: str | None = None, **h):
        set_seed(seed)
        self.h = {**self.HP, **h}
        if device is None or device.lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()

        self.model: Optional[_SASRec] = None
        self.item2idx, self.idx2item = {}, []
        self.max_price = 1.0

    # ─────────────────────── preprocessing ────────────────────────
    def _ensure_schema(self, log: pd.DataFrame,
                       item_feats: Optional[pd.DataFrame]):
        rename = {"user_idx": "user_id", "item_idx": "item_id"}
        log = log.rename(columns={k: v for k,v in rename.items() if k in log.columns})
        if item_feats is not None:
            item_feats = item_feats.rename(columns={k: v for k,v in rename.items()
                                                     if k in item_feats.columns})
        if "price" not in log.columns:
            if item_feats is not None and "price" in item_feats.columns:
                m = item_feats.set_index("item_id")["price"].to_dict()
                log["price"] = log["item_id"].map(m).fillna(1.0)
            else:
                log["price"] = 1.0
        if "timestamp" not in log.columns:
            log = log.reset_index(drop=True); log["timestamp"] = log.index
        req = {"user_id","item_id","timestamp","price"}
        if missing := (req - set(log.columns)):
            raise ValueError(f"log missing {missing}")
        return log, item_feats

    def _build_vocab(self, log: pd.DataFrame):
        self.idx2item = [0] + log["item_id"].unique().tolist()
        self.item2idx = {it: idx for idx,it in enumerate(self.idx2item)}
        self.max_price = log["price"].max()

    def _user_seqs(self, log: pd.DataFrame):
        log = log.sort_values(["user_id","timestamp"])
        seqs = defaultdict(list)
        for u,it in zip(log["user_id"], log["item_id"]):
            seqs[u].append(self.item2idx[it])
        return seqs

    # ───────────────────────── training ───────────────────────────
    def fit(self, log, user_features=None, item_features=None):
        log_pd = to_pandas(log)
        item_pd = to_pandas(item_features)
        log_pd, item_pd = self._ensure_schema(log_pd, item_pd)

        self._build_vocab(log_pd)
        seqs = self._user_seqs(log_pd)
        price_map = log_pd.drop_duplicates("item_id") \
                          .set_index("item_id")["price"].to_dict()

        ds = _SeqDataset(seqs, price_map, list(self.item2idx.values()),
                         self.h["max_len"])
        dl = DataLoader(ds, batch_size=self.h["batch"], shuffle=True, num_workers=0)

        self.model = _SASRec(len(self.idx2item)-1, self.h["dim"],
                             self.h["layers"], self.h["heads"],
                             self.h["dropout"], self.h["max_len"]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.h["lr"], weight_decay=self.h["wd"])

        best, stall = float("inf"), 0
        for ep in range(1, self.h["epochs"]+1):
            self.model.train(); epoch_loss = 0.0
            for seq,pos,neg,price in dl:
                seq,pos,neg,price = (t.to(self.device)
                                     for t in (seq,pos,neg,price))
                price = price / self.max_price
                h = self.model(seq)[:, -1, :]
                lp = (h * self.model.item_emb(pos)).sum(-1)
                ln = (h * self.model.item_emb(neg)).sum(-1)
                loss = -(price * torch.log(torch.sigmoid(lp-ln)+1e-8)).mean()

                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()*seq.size(0)
            epoch_loss /= len(ds)
            if epoch_loss < best-1e-5: best,stall = epoch_loss,0
            else: stall += 1;  # early stop
            if stall >= self.h["patience"]: break
        self.model.eval()

    # ──────────────────────── inference ────────────────────────────
    def _score_batch(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        h = self.model(seq_tensor.to(self.device))[:, -1, :]
        return h @ self.model.item_emb.weight.T              

    def predict(self, log, k, users, items, user_features=None,
                item_features=None, filter_seen_items: bool=True):
        if SparkSession is None:
            raise ImportError("PySpark required for predict()")
        spark = SparkSession.builder.getOrCreate()

        log_pd = to_pandas(log)
        users_pd = to_pandas(users)[["user_idx"]].rename(columns={"user_idx":"user_id"})
        items_pd = to_pandas(items)[["item_idx"]].rename(columns={"item_idx":"item_id"})
        item_pd  = to_pandas(item_features)
        log_pd, item_pd = self._ensure_schema(log_pd, item_pd)

        # price map
        if item_pd is not None and "price" in item_pd.columns:
            price_map = item_pd.set_index("item_id")["price"].to_dict()
        else:
            price_map = log_pd.drop_duplicates("item_id") \
                              .set_index("item_id")["price"].to_dict()

        seqs = self._user_seqs(log_pd)
        seen = defaultdict(set)
        if filter_seen_items:
            for u,it in zip(log_pd["user_id"], log_pd["item_id"]):
                seen[u].add(it)

        results, batch_size = [], 512
        user_ids = users_pd["user_id"].tolist()
        for b in range(0,len(user_ids),batch_size):
            batch_users = user_ids[b:b+batch_size]
            seq_list=[]
            for u in batch_users:
                s = seqs.get(u, [])[-self.h["max_len"]:]
                seq_list.append([0]*(self.h["max_len"]-len(s))+s)
            seq_tensor = torch.tensor(seq_list,dtype=torch.long)

            with torch.no_grad():
                logits = self._score_batch(seq_tensor).cpu()
            probs = torch.sigmoid(logits).numpy()             

            for i,u in enumerate(batch_users):
                scores = probs[i]                         
                scores[0] = -np.inf                            
                # expected revenue = prob × price**exp
                prices_arr = np.array(
                    [price_map.get(self.idx2item[idx],1.0)
                     for idx in range(len(self.idx2item))],
                    dtype=np.float32)
                exp_rev = scores * (prices_arr ** self.h["price_exp"])

                if filter_seen_items and u in seen:
                    for it in seen[u]:
                        exp_rev[self.item2idx.get(it,0)] = -np.inf

                top_idx = np.argpartition(-exp_rev, range(k))[:k]
                top_idx = top_idx[np.argsort(-exp_rev[top_idx])]

                for idx in top_idx:
                    results.append((int(u),
                                    int(self.idx2item[idx]),
                                    float(exp_rev[idx])))

        return spark.createDataFrame(results,
                                     schema=["user_idx","item_idx","relevance"])

