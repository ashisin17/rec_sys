"""
GradientBoostRecommender 
Ashita Singh
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

# ── Spark glue ───────────────────────────────────────────────
try:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window, DataFrame as SparkDF
except ImportError:  # local / non‑Spark
    sf = Window = SparkDF = None  # type: ignore

try:
    from sim4rec.utils import pandas_to_spark
except ImportError:
    def pandas_to_spark(df: pd.DataFrame):
        return df

# ── helper utilities ────────────────────────────────────────────────────

def _high_card(col: pd.Series, th: int = 30) -> bool:
    return col.dtype == "object" and col.nunique() >= th

def _target_encode(series: pd.Series, target: pd.Series, global_mean: float, prior: float = 10):
    stats = target.groupby(series).agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + prior * global_mean) / (stats["count"] + prior)
    return series.map(smooth).astype(np.float32)

def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in df.columns if df[c].dtype == "object" and not _high_card(df[c])]
    return pd.get_dummies(df, columns=cats, dummy_na=False) if cats else df

def _standardise(df: pd.DataFrame, cols: List[str]):
    if not cols:
        return None, None
    mean_ = df[cols].mean().values
    std_ = df[cols].std(ddof=0).replace(0, 1).values
    df[cols] = (df[cols] - mean_) / std_
    return mean_, std_

# ── main model ─────────────────────────────────────────────────────────

class GradientBoostRecommender:
    """LightGBM LambdaRank recommender optimised for discounted revenue."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        price_pow: float | None = None,
        edr_alpha: float | None = None,
        lgb_params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 2500,
        early_stopping_rounds: int = 200,
        auto_tune: bool = False,
    ) -> None:
        self.seed = seed
        self.price_pow = price_pow or 1.4
        self.edr_alpha = edr_alpha or 0.8
        self.auto_tune = auto_tune

        self.model: lgb.Booster | None = None
        self.cols: List[str] = []
        self.num_cols: List[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.user_stats: pd.DataFrame | None = None
        self.item_stats: pd.DataFrame | None = None

        base = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            label_gain=[0, 1],
            learning_rate=0.05,
            num_leaves=48,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=15,
            lambda_l2=5.0,
            random_state=seed,
            verbosity=-1,
        )
        if lgb_params:
            base.update(lgb_params)
        self.base_params = base
        self.num_boost_round = num_boost_round
        self.early = early_stopping_rounds

    # ── aggregate features ──
    @staticmethod
    def _build_user_stats(df: pd.DataFrame):
        g = df.groupby("user_idx")
        return pd.DataFrame({
            "user_idx": g.size().index,
            "user_price_mean": g["price"].mean().values,
            "user_ctr": g["relevance"].mean().values,
        })

    @staticmethod
    def _build_item_stats(df: pd.DataFrame):
        g = df.groupby("item_idx")
        return pd.DataFrame({
            "item_idx": g.size().index,
            "item_buy_rate": g["relevance"].mean().values,
            "item_log_views": np.log1p(g.size().values),
        })

    # ── dataframe construction ──
    def _prepare_dataframe(self, log, user_features, item_features):
        pdf = log.join(user_features, "user_idx").join(item_features, "item_idx").toPandas()
        self.user_stats = self._build_user_stats(pdf)
        self.item_stats = self._build_item_stats(pdf)
        pdf = pdf.merge(self.user_stats, on="user_idx", how="left").merge(self.item_stats, on="item_idx", how="left")
        pdf = pdf.sort_values("user_idx").reset_index(drop=True)

        y = pdf["relevance"].astype(int)
        global_mean = y.mean()
        for c in pdf.columns:
            if _high_card(pdf[c]):
                pdf[c] = _target_encode(pdf[c], y, global_mean)
        pdf = _one_hot(pdf)
        pdf["price_log"] = np.log1p(pdf["price"])
        pdf["price_sqrt"] = np.sqrt(pdf["price"])

        X = pdf.drop(columns=[c for c in ("relevance", "user_idx", "item_idx", "__iter") if c in pdf.columns])
        self.num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number) and X[c].nunique() > 10]
        self.mean_, self.std_ = _standardise(X, self.num_cols)
        self.cols = X.columns.tolist()
        groups = pdf.groupby("user_idx").size().values.tolist()
        return X, y, groups

    # ── training ──
    def _train_booster(self, X, y, groups, price_pow, params):
        sample_w = (np.nan_to_num(X["price"].values) ** price_pow).astype(np.float32)
        dtrain = lgb.Dataset(X, y, weight=sample_w, group=groups, free_raw_data=False)
        return lgb.train(params, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dtrain],
                         callbacks=[lgb.early_stopping(self.early, verbose=False), lgb.log_evaluation(period=0)])

    def _pseudo_rev(self, booster, X, alpha):
        pred = booster.predict(X, num_iteration=booster.best_iteration)
        price_arr = np.nan_to_num(X["price"].values)
        return np.nanmean(pred * (price_arr ** alpha))

    def fit(self, log, *, user_features=None, item_features=None):
        if None in (log, user_features, item_features):
            raise ValueError("log, user_features, item_features required")
        X, y, groups = self._prepare_dataframe(log, user_features, item_features)

        if self.auto_tune:
            grid_pp = [1.1, 1.3, 1.5, 1.7]
            grid_alpha = [0.75, 0.8, 0.85, 0.9]
            grid_lr = [0.04, 0.05, 0.06]
            best, cfg, bst = -np.inf, None, None
            for pp, a, lr in itertools.product(grid_pp, grid_alpha, grid_lr):
                p = self.base_params.copy(); p["learning_rate"] = lr
                m = self._train_booster(X, y, groups, pp, p)
                score = self._pseudo_rev(m, X, a)
                if np.isnan(score):
                    continue
                if score > best:
                    best, cfg, bst = score, (pp, a, lr), m
            if cfg is None:
                raise RuntimeError("auto_tune failed: all combos NaN")
            self.price_pow, self.edr_alpha, self.base_params["learning_rate"] = cfg
            self.model = bst
        else:
            self.model = self._train_booster(X, y, groups, self.price_pow, self.base_params)
        return self

    # ── cross-table for prediction ──
    def _prepare_cross(self, users, items, user_features, item_features, log=None, filter_seen=True):
        if hasattr(users, "toPandas"):
            cross = users.join(items).drop("__iter").toPandas()\
                         .merge(user_features.toPandas(), on="user_idx")\
                         .merge(item_features.toPandas(), on="item_idx")
            if filter_seen and log is not None and log.count():
                seen = log.select("user_idx", "item_idx").toPandas()
                cross = cross.merge(seen.assign(_s=1), on=["user_idx", "item_idx"], how="left")
                cross = cross[cross["_s"].isna()].drop(columns="_s")
        else:
            cross = users.merge(items, how="cross").merge(user_features, on="user_idx").merge(item_features, on="item_idx")
            if filter_seen and log is not None:
                cross = cross.merge(log.assign(_s=1), on=["user_idx", "item_idx"], how="left")
                cross = cross[cross["_s"].isna()].drop(columns=["_s", "relevance"], errors="ignore")
        return cross

    # ── predict ──
    def predict(self, log, k, users, items, *, user_features=None, item_features=None, filter_seen_items=True):
        if self.model is None:
            raise RuntimeError("fit() not called")
        cross = self._prepare_cross(users, items, user_features, item_features, log, filter_seen_items)
        cross = cross.merge(self.user_stats, on="user_idx", how="left").merge(self.item_stats, on="item_idx", how="left")
        cross = cross.fillna({"user_price_mean": 0, "user_ctr": 0, "item_buy_rate": 0, "item_log_views": 0})

        # encode cats same way
        for c in cross.columns:
            if _high_card(cross[c]):
                cross[c] = 0.0
        cross = _one_hot(cross)
        for c in self.cols:
            if c not in cross.columns:
                cross[c] = 0
        Xc = cross[self.cols].copy()
        if self.num_cols:
            common = [c for c in self.num_cols if c in Xc.columns]
            idx = [self.num_cols.index(c) for c in common]
            Xc[common] = (Xc[common] - self.mean_[idx]) / self.std_[idx]

        cross["score"] = self.model.predict(Xc, num_iteration=self.model.best_iteration)
        # expected discounted revenue proxy
        cross["edr"] = cross["score"] * (cross["price"].values ** self.edr_alpha)

        cross = cross.sort_values(["user_idx", "edr"], ascending=[True, False])
        topk = cross.groupby("user_idx").head(k)

        recs_pd = topk[["user_idx", "item_idx"]].assign(relevance=topk["edr"])
        if SparkDF is not None and hasattr(users, "_jdf"):
            recs = pandas_to_spark(recs_pd)
            w = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
            return recs.withColumn("rank", sf.row_number().over(w)).filter(sf.col("rank") <= k).drop("rank")
        return recs_pd
