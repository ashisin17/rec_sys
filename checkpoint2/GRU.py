import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 3g pyspark-shell"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os, sys

# tell Spark EXACTLY which python to launch => added cause spark/python issues for windows :(
os.environ["PYSPARK_PYTHON"]        = sys.executable   # workers
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable   # driver

from pyspark.sql import SparkSession


# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.driver.maxResultSize", "1g") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.python.worker.reuse", "false") \
    .config("spark.python.worker.connectionTimeout", "600")\
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender, 
    SVMRecommender, 
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Cell: Define custom recommender template
"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""
# Checkpoint 1: Random Forest! 

#needed imports!!!!!!
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split


class MyRecommender:

    # Initialize hyperr params! 
    def __init__(self,
                 seed: int | None = None,
                 n_estimators: int = 400,
                 max_depth: int | None = 12,
                 min_samples_leaf: int = 3,
                 tune_hyper: bool = False           # set True ➜ 3fold CV
                 ):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tune_hyper = tune_hyper
        self.pipeline: Pipeline | None = None      # sklearn pipeline stub

        self._feature_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []

    # training!
    def fit(self,
            log: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame,
            ) -> None:
        

        # spark => pandas
        log_pd   = log.select("user_idx", "item_idx", "relevance").toPandas()
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # join so we can get full feature row per interaction 
        full_df = (log_pd
                   .merge(users_pd,  on="user_idx")
                   .merge(items_pd,  on="item_idx"))

        # make sure label is binary!!!
        full_df["label"] = (full_df["relevance"] > 0).astype(int)

        # feature lists
        user_num = [c for c in users_pd.columns if c.startswith("user_attr_")]
        item_num = [c for c in items_pd.columns if c.startswith("item_attr_")]
        self._num_cols = user_num + item_num + ["price"]          # scale these

        self._cat_cols = ["segment", "category"]                  # one-hot
        self._feature_cols = self._num_cols + self._cat_cols

        X = full_df[self._feature_cols]
        y = full_df["label"]

        # slight undersampling of majority (0) class
        pos_idx = full_df.index[full_df["label"] == 1]

        neg_idx = (
            full_df[full_df["label"] == 0]                       # work on DF, not Index!!!!!!!!
                .sample(frac=min(1.0,
                                len(pos_idx) /
                                max(1, len(full_df) - len(pos_idx))),
                        random_state=self.seed)
                .index
        )

        sample_df = full_df.loc[pos_idx.union(neg_idx)]
        X = sample_df[self._feature_cols]
        y = sample_df["label"]

        # preprocessing + model pipeline 
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self._num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self._cat_cols)
            ])

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced_subsample"      # ≈ L2 regularisation
        )

        pipe = Pipeline([
            ("prep", preprocess),
            ("model", rf)
        ])


        if self.tune_hyper:
            param_grid = {
                "model__max_depth": [6, 12, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__n_estimators": [200, 400],
            }
            pipe = GridSearchCV(pipe, param_grid, cv=3,
                                scoring="roc_auc", n_jobs=-1, verbose=0)

        pipe.fit(X, y)
        self.pipeline = pipe             

    # preditcion + ranking!
    def predict(self,
                log: DataFrame,
                k: int,
                users: DataFrame,
                items: DataFrame,
                user_features: DataFrame | None = None,
                item_features: DataFrame | None = None,
                filter_seen_items: bool = True
                ) -> DataFrame:
        

        assert self.pipeline is not None, "call .fit() before .predict()"

        # spark -> pandas
        users_pd  = users.toPandas()
        items_pd  = items.toPandas()

        seen_pairs: set[tuple[int, int]] = set()
        if filter_seen_items and log is not None:
            seen_pairs = set(log.select("user_idx", "item_idx").toPandas()
                             .itertuples(index=False, name=None))

        users_pd["_tmp"] = 1
        items_pd["_tmp"] = 1
        cand = users_pd.merge(items_pd, on="_tmp").drop("_tmp", axis=1)

        # remove seen items
        if seen_pairs:
            mask = ~cand.apply(lambda r: (r.user_idx, r.item_idx) in seen_pairs,
                               axis=1)
            cand = cand[mask]

        # prob + expected revenue 
        X_pred = cand[self._feature_cols]
        prob   = self.pipeline.predict_proba(X_pred)[:, 1]        # P(click)
        cand["relevance"] = prob * cand["price"]                  

        # top kper user 
        cand.sort_values(["user_idx", "relevance"], ascending=[True, False],
                         inplace=True)
        cand["rank"] = cand.groupby("user_idx").cumcount() + 1
        cand = cand[cand["rank"] <= k]

        # back to Spark!
        recs_df = spark.createDataFrame(cand[["user_idx", "item_idx",
                                              "relevance"]])

        return recs_df

#Checkpoint 2: GRU Recommender!

#handle all needed inputs!
import torch, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict

#make sure we have time stamp col
def _ensure_ts(pdf: pd.DataFrame) -> pd.DataFrame:
    if "__ts" not in pdf.columns:
        pdf["__ts"] = (
            pdf.get("timestamp")
            or pdf.get("__iter")
            or np.arange(len(pdf))
        )
    return pdf

# dataset for next item predd
class _SequenceDataset(Dataset):
    def __init__(self, sequences, targets, pad_idx, max_len):
        self.seq, self.tgt = sequences, targets
        self.pad, self.maxL = pad_idx, max_len

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        hist = self.seq[idx][-self.maxL :]
        return (
            torch.tensor(hist, dtype=torch.long),
            len(hist),
            torch.tensor(self.tgt[idx], dtype=torch.long),
        )

    # instance method ⇒ has access to self.pad
    def collate(self, batch):
        seqs, lens, tgts = zip(*batch)
        return (
            pad_sequence(seqs, True, self.pad),
            torch.tensor(lens),
            torch.stack(tgts),
        )

# GRU backbone! 
class _GRURec(nn.Module):
    def __init__(self, n_items, embed_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, seq, lengths):
        x = self.emb(seq)
        packed = pack_padded_sequence(x, lengths.cpu(), True, False)
        _, h = self.gru(packed)          
        z = self.proj(h[-1])             # last layer’s hidden state
        return torch.matmul(z, self.emb.weight.t())  # (B, |V|)


class GRURecommender:
    def __init__(
        self,
        seed: int = 42,
        embed_dim: int = 64, #64 embed + 128 hidden combo work best => higher leads to worse performance :(
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        epochs: int = 5, 
        max_seq_len: int = 50,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # hyperparams
        self.embed_dim, self.hidden_dim = embed_dim, hidden_dim
        self.n_layers, self.dropout = n_layers, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs = batch_size, epochs
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # handle state
        self.model: _GRURec | None = None
        self.n_items: int | None = None
        self.pad_idx: int | None = None
        self.price_vec: torch.Tensor | None = None
        self._user_hist: dict[int, list[tuple]] = defaultdict(list)


    def _append_histories(self, log_pd: pd.DataFrame, price_lookup: dict[int, float]):
        """
        Extend self._user_hist with *new* rows from log_pd.
        Assumes log_pd is already timestamp-sorted.
        """
        for uid, ts, iid, rel in log_pd[
            ["user_idx", "__ts", "item_idx", "relevance"]
        ].itertuples(False):
            tup = (ts, iid, float(price_lookup[iid]), int(rel > 0))
            self._user_hist[uid].append(tup)

            # soft cap => keep at most 3×max_seq_len for memory
            if len(self._user_hist[uid]) > self.max_seq_len * 3:
                self._user_hist[uid] = self._user_hist[uid][-self.max_seq_len * 3 :]

    #retrain on a new log slice 
    def fit(self, log, user_features, item_features):
        # pandas prep 
        log_pd = _ensure_ts(
            log.select("user_idx", "item_idx", "relevance").toPandas()
        )
        log_pd.sort_values(["user_idx", "__ts"], inplace=True)

        items_pd = item_features.select("item_idx", "price").toPandas()
        price_lookup = dict(zip(items_pd.item_idx, items_pd.price))

        if self.n_items is None:                 
            prices = items_pd.sort_values("item_idx")["price"].values
            self.n_items = len(prices)
            self.pad_idx = self.n_items           
            self.price_vec = torch.tensor(
                np.append(prices, 0.0), dtype=torch.float32, device=self.device
            )

        # extend histories w/ the new slice 
        self._append_histories(log_pd, price_lookup)

        # build training examples just from new rows!
        seq_ids, tgts = [], []
        for uid, grp in log_pd.groupby("user_idx", sort=False):
            full_items = [t[1] for t in self._user_hist[uid]]
            fresh_item_pos = [i for i, t in enumerate(self._user_hist[uid]) if t in grp.apply(lambda r: (r["__ts"], r["item_idx"], float(price_lookup[r["item_idx"]]), int(r["relevance"]>0)), axis=1).tolist()]
            for pos in fresh_item_pos:
                if pos == 0:
                    continue
                prefix = full_items[max(0, pos - self.max_seq_len) : pos]
                seq_ids.append(prefix)
                tgts.append(full_items[pos])

        if not seq_ids:                # nothing new → nothing to train
            return

        # dataset + dataloader 
        ds = _SequenceDataset(seq_ids, tgts, self.pad_idx, self.max_seq_len)
        dl = DataLoader(ds, self.batch_size, True, collate_fn=ds.collate)

        # (re)build model on first call ONLYYY
        if self.model is None:
            self.model = _GRURec(
                self.n_items + 1,
                self.embed_dim,
                self.hidden_dim,
                self.n_layers,
                self.dropout,
                self.pad_idx,
            ).to(self.device)

        # training loop 
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for ep in range(self.epochs):
            tot_loss = 0.0
            for seq_pad, lens, tgt in dl:
                seq_pad, lens, tgt = (
                    seq_pad.to(self.device),
                    lens.to(self.device),
                    tgt.to(self.device),
                )
                opt.zero_grad()
                logits = self.model(seq_pad, lens)
                loss = loss_fn(logits, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot_loss += loss.item() * len(tgt)
            print(
                f"[GRU] epoch {ep+1}/{self.epochs} – "
                f"loss={tot_loss/len(ds):.4f}"
            )

        self.model.eval()  # inference mode!

    # inference 
    @torch.no_grad()
    def predict(
        self,
        log,
        k,
        users,
        items,
        user_features=None,
        item_features=None,
        filter_seen_items: bool = True,
    ):
        assert self.model is not None, "call fit() first"

        uid_list = users.select("user_idx").toPandas()["user_idx"].tolist()

        seq_tensors, seq_lens = [], []
        for uid in uid_list:
            item_ids = [t[1] for t in self._user_hist.get(uid, [])]
            if not item_ids:
                seq = torch.tensor([self.pad_idx])
            else:
                seq = torch.tensor(item_ids[-self.max_seq_len :])
            seq_tensors.append(seq)
            seq_lens.append(len(seq))

        seq_pad = pad_sequence(seq_tensors, True, self.pad_idx).to(self.device)
        seq_len = torch.tensor(seq_lens).to(self.device)

        logits = self.model(seq_pad, seq_len)[:, : self.n_items]
        probs = torch.softmax(logits, dim=-1)
        exp_rev = probs * self.price_vec[: self.n_items]

        if filter_seen_items and log is not None:
            seen = set(
                log.select("user_idx", "item_idx")
                .toPandas()
                .itertuples(False, None)
            )
        else:
            seen = set()

        rows = []
        for b, uid in enumerate(uid_list):
            scores = exp_rev[b].cpu().numpy()
            if filter_seen_items:
                for t in self._user_hist.get(uid, []):
                    scores[t[1]] = -np.inf
            for it in scores.argsort()[-k:][::-1]:
                rows.append((uid, int(it), float(scores[it])))

        return spark.createDataFrame(rows, ["user_idx", "item_idx", "relevance"])


# Cell: Data Exploration Functions
"""
## Data Exploration Functions
These functions help us understand the generated synthetic data.
"""

def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        print("User feature correlations saved to 'user_feature_correlations.png'")


def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        print("Item feature distributions saved to 'item_feature_distributions.png'")


def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = history_df.join(
            users_df.select('user_idx', 'segment'),
            on='user_idx'
        ).join(
            items_df.select('item_idx', 'category'),
            on='item_idx'
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        print("Segment-category interactions saved to 'segment_category_interactions.png'")


# Cell: Recommender Analysis Function
"""
## Recommender System Analysis
This is the main function to run analysis of different recommender systems and visualize the results.
"""

def run_recommender_analysis():
    """
    Run an analysis of different recommender systems and visualize the results.
    This function creates a synthetic dataset, performs EDA, evaluates multiple recommendation
    algorithms using train-test split, and visualizes the performance metrics.
    """
    # Create a smaller dataset for experimentation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000  # Reduced from 10,000
    config['data_generation']['n_items'] = 200   # Reduced from 1,000
    config['data_generation']['seed'] = 42       # Fixed seed for reproducibility
    
    # Get train-test split parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate user data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    # Generate item data
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    # Generate initial interaction history
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Cell: Exploratory Data Analysis
    """
    ## Exploratory Data Analysis
    Let's explore the generated synthetic data before running the recommenders.
    """
    
    # Perform exploratory data analysis on the generated data
    print("\n=== Starting Exploratory Data Analysis ===")
    explore_user_data(users_df)
    explore_item_data(items_df)
    explore_interactions(history_df, users_df, items_df)
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Cell: Setup and Run Recommenders
    """
    ## Recommender Systems Comparison
    Now we'll set up and evaluate different recommendation algorithms.
    """
    
    # Initialize recommenders to compare
    #6/2 Note: commenting out 2 code blocks below to speed up entire process, just wanna test MY Recommenders! 

    #recommenders = [
     #   SVMRecommender(seed=42), 
      #  RandomRecommender(seed=42),
       # PopularityRecommender(alpha=1.0, seed=42),
        #ContentBasedRecommender(similarity_threshold=0.0, seed=42),
        #MyRecommender(seed=42)  # Add your custom recommender here
    #]
    #recommender_names = ["SVM", "Random", "Popularity", "ContentBased", "MyRecommender"]

    # temp evaluate only the new Random-Forest model!
    #recommenders = [MyRecommender(seed=42, tune_hyper=False)]
    #recommender_names = ["MyRecommender"]

     #=== temp eval on Seq Rec only ===
    #recommenders      = [MySeqRecommender()]
    #recommender_names = ["MySeqRecommender"]

    recommenders = [
        GRURecommender()
    ]

    recommender_names = ["GRURecommender"]


    
    # Initialize recommenders with initial history
    for recommender in recommenders:
        recommender.fit(log=data_generator.history_df, 
                        user_features=users_df, 
                        item_features=items_df)
    
    # Evaluate each recommender separately using train-test split
    results = []
    
    for name, recommender in zip(recommender_names, recommenders):
        print(f"\nEvaluating {name}:")
        
        # Clean up any existing simulator data directory for this recommender
        simulator_data_dir = f"simulator_train_test_data_{name}"
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
            print(f"Removed existing simulator data directory: {simulator_data_dir}")
        
        # Initialize simulator
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=simulator_data_dir,
            log_df=data_generator.history_df,  # PySpark DataFrames don't have copy method
            conversion_noise_mean=config['simulation']['conversion_noise_mean'],
            conversion_noise_std=config['simulation']['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Run simulation with train-test split
        train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
            recommender=recommender,
            train_iterations=train_iterations,
            test_iterations=test_iterations,
            user_frac=config['simulation']['user_fraction'],
            k=config['simulation']['k'],
            filter_seen_items=config['simulation']['filter_seen_items'],
            retrain=config['simulation']['retrain']
        )
        
        # Calculate average metrics
        train_avg_metrics = {}
        for metric_name in train_metrics[0].keys():
            values = [metrics[metric_name] for metrics in train_metrics]
            train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
        
        test_avg_metrics = {}
        for metric_name in test_metrics[0].keys():
            values = [metrics[metric_name] for metrics in test_metrics]
            test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
        
        # Store results
        results.append({
            "name": name,
            "train_total_revenue": sum(train_revenue),
            "test_total_revenue": sum(test_revenue),
            "train_avg_revenue": np.mean(train_revenue),
            "test_avg_revenue": np.mean(test_revenue),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_revenue": train_revenue,
            "test_revenue": test_revenue,
            **train_avg_metrics,
            **test_avg_metrics
        })
        
        # Print summary for this recommender
        print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
        print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
        performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
        print(f"  Performance Change: {performance_change:.2f}%")
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
    
    # Print summary table
    print("\nRecommender Evaluation Results (sorted by test revenue):")
    summary_cols = ["name", "train_total_revenue", "test_total_revenue", 
                   "train_avg_revenue", "test_avg_revenue",
                   "train_precision_at_k", "test_precision_at_k",
                   "train_ndcg_at_k", "test_ndcg_at_k",
                   "train_mrr", "test_mrr",
                   "train_discounted_revenue", "test_discounted_revenue"]
    summary_cols = [col for col in summary_cols if col in results_df.columns]
    
    print(results_df[summary_cols].to_string(index=False))
    
    # Cell: Results Visualization
    """
    ## Results Visualization
    Now we'll visualize the performance of the different recommenders.
    """
    
    # Generate comparison plots
    visualize_recommender_performance(results_df, recommender_names)
    
    # Generate detailed metrics visualizations
    visualize_detailed_metrics(results_df, recommender_names)
    
    return results_df


# Cell: Performance Visualization Functions
"""
## Performance Visualization Functions
These functions create visualizations for comparing recommender performance.
"""

def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_discounted_revenue' in results_df.columns and 'test_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue']
        test_revenue = results_df.iloc[i]['test_revenue']
        
        # Check if revenue is a scalar (numpy.float64) or a list/array
        if isinstance(train_revenue, (float, np.float64, np.float32, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, np.float32, int, np.integer)):
            test_revenue = [test_revenue]
            
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'train_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'train_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    # Get best-performing model
    best_model_idx = results_df['test_total_revenue'].idxmax()
    best_model_name = results_df.iloc[best_model_idx]['name']
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'test_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")


def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics
    all_metrics = []
    if len(results_df) > 0 and 'train_metrics' in results_df.columns:
        first_train_metrics = results_df.iloc[0]['train_metrics'][0]
        all_metrics = list(first_train_metrics.keys())
    
    # Select key metrics to visualize
    key_metrics = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric in enumerate(key_metrics):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name].iloc[0]
                
                # Get metric values for training phase
                train_values = []
                for train_metric in row['train_metrics']:
                    if metric in train_metric:
                        train_values.append(train_metric[metric])
                
                # Get metric values for testing phase
                test_values = []
                for test_metric in row['test_metrics']:
                    if metric in test_metric:
                        test_values.append(test_metric[metric])
                
                # Plot training phase
                plt.plot(range(len(train_values)), train_values, 
                         marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='-', label=f"{name} (train)")
                
                # Plot testing phase
                plt.plot(range(len(train_values), len(train_values) + len(test_values)), 
                         test_values, marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='--', label=f"{name} (test)")
                
                # Add a vertical line to separate train and test
                if j == 0:  # Only add the line once
                    plt.axvline(x=len(train_values)-0.5, color='k', 
                                linestyle='--', alpha=0.3, label='Train/Test Split')
            
            # Get metric info from EVALUATION_METRICS
            if metric in EVALUATION_METRICS:
                metric_info = EVALUATION_METRICS[metric]
                metric_name = metric_info['name']
                plt.title(f"{metric_name} Trajectory")
            else:
                plt.title(f"{metric.replace('_', ' ').title()} Trajectory")
            
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            
            # Add legend to the last plot only to avoid cluttering
            if i == len(key_metrics) - 1 or i == 5:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('recommender_metrics_trajectories.png')
    print("Detailed metrics visualizations saved to 'recommender_metrics_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns
    metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols = [col for col in metric_cols if not col.endswith('_metrics') and not col.endswith('_revenue')]
    
    if len(metric_cols) > 1:
        correlation_df = results_df[metric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig('metrics_correlation_heatmap.png')
        print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")


def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
        
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    for user_id, user_recs in recommendations.groupBy("user_idx").agg(
        sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations")
    ).collect():
        # Sort by rank
        user_rec_list = sorted(user_id.recommendations, key=lambda x: x[1])
        
        # Calculate DCG
        dcg = 0
        for i, (rel, _) in enumerate(user_rec_list[:k]):
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        
        user_dcg.append(dcg)
    
    # Return average DCG across all users
    return np.mean(user_dcg) if user_dcg else 0.0


# Cell: Main execution
"""
## Run the Analysis
When you run this notebook, it will perform the full analysis and visualization.
"""

if __name__ == "__main__":
    results = run_recommender_analysis() 