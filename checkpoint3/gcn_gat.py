import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling
import numpy as np
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql.window import Window


class GCNRecommender:
    """Graph Convolutional Network Recommender"""
    
    def __init__(self, embedding_dim=128, num_layers=4, dropout=0.2, l2_lambda=0.02, seed=None):
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_graph(self, log, users, items):
        log_pd = log.toPandas()
        users_pd = users.toPandas()
        items_pd = items.toPandas()
        
        user_mapping = {u: i for i, u in enumerate(users_pd['user_idx'])}
        item_mapping = {i: j+len(user_mapping) for j, i in enumerate(items_pd['item_idx'])}
        
        edge_index = torch.tensor([
            [user_mapping[u] for u in log_pd['user_idx']],
            [item_mapping[i] for i in log_pd['item_idx']]
        ], dtype=torch.long)
        
        edge_weight = torch.tensor(log_pd['relevance'].values, dtype=torch.float)
        
        data = HeteroData()
        data['user'].node_id = torch.arange(len(user_mapping))
        data['item'].node_id = torch.arange(len(item_mapping))
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_weight = edge_weight
        
        return data.to(self.device)

    def fit(self, log, user_features=None, item_features=None):
        self.graph = self._build_graph(log, user_features, item_features)
        
        self.model = GCN(
            num_users=len(self.graph['user'].node_id),
            num_items=len(self.graph['item'].node_id),
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=self.l2_lambda)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(100):
            self.model.train()
            optimizer.zero_grad()
            
            pos_pred = self.model(self.graph)
            neg_edge_index = negative_sampling(
                edge_index=self.graph['user', 'interacts', 'item'].edge_index,
                num_nodes=(len(self.graph['user'].node_id), len(self.graph['item'].node_id)),
                num_neg_samples=self.graph['user', 'interacts', 'item'].edge_index.size(1)
            )
            
            neg_pred = self.model(self.graph, neg_edge_index)
            loss = criterion(pos_pred, torch.ones_like(pos_pred)) + \
                   criterion(neg_pred, torch.zeros_like(neg_pred))
            
            loss.backward()
            optimizer.step()

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate top-k recommendations for each user.
        Returns a Spark DataFrame with columns: user_idx, item_idx, relevance
        """
        self.model.eval()
        with torch.no_grad():
            users_pd = users.toPandas()
            items_pd = items.toPandas()
            user_ids = users_pd['user_idx'].values
            item_ids = items_pd['item_idx'].values

            user_emb = self.model.user_emb.weight
            item_emb = self.model.item_emb.weight
            
            scores = torch.matmul(user_emb, item_emb.T).cpu().numpy()
            
            recs = []
            for i, u in enumerate(user_ids):
                user_scores = scores[i]
                topk_idx = np.argsort(-user_scores)[:k]
                for idx in topk_idx:
                    recs.append((u, item_ids[idx], float(user_scores[idx])))
            
            recs_df = pd.DataFrame(recs, columns=['user_idx', 'item_idx', 'relevance'])
            
            if filter_seen_items and log is not None:
                log_pd = log.toPandas()
                seen = set(zip(log_pd['user_idx'], log_pd['item_idx']))
                recs_df = recs_df[~recs_df.set_index(['user_idx', 'item_idx']).index.isin(seen)]
            
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            recommendations_df = spark.createDataFrame(recs_df)
            return recommendations_df


class GATRecommender:
    """Graph Attention Network Recommender"""

    def __init__(self, embedding_dim=128, heads=4, dropout=0.2, l2_lambda=0.01, seed=None):
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_graph(self, log, users, items):
        log_pd = log.toPandas()
        users_pd = users.toPandas()
        items_pd = items.toPandas()

        user_ids = users_pd['user_idx'].unique()
        item_ids = items_pd['item_idx'].unique()
        self.user2id = {u: i for i, u in enumerate(user_ids)}
        self.item2id = {i: j for j, i in enumerate(item_ids)}
        self.id2user = {i: u for u, i in self.user2id.items()}
        self.id2item = {j: i for i, j in self.item2id.items()}

        user_indices = log_pd['user_idx'].map(self.user2id)
        item_indices = log_pd['item_idx'].map(self.item2id)
        edge_index = torch.tensor([user_indices.values, item_indices.values + len(self.user2id)], dtype=torch.long)
        edge_weight = torch.tensor(log_pd['relevance'].values, dtype=torch.float)

        data = HeteroData()
        data['user'].num_nodes = len(self.user2id)
        data['item'].num_nodes = len(self.item2id)
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_weight = edge_weight

        return data.to(self.device)

    def fit(self, log, user_features=None, item_features=None):
        self.graph = self._build_graph(log, user_features, item_features)
        num_users = self.graph['user'].num_nodes
        num_items = self.graph['item'].num_nodes
        edge_index = self.graph['user', 'interacts', 'item'].edge_index

        self.user_emb = nn.Embedding(num_users, self.embedding_dim).to(self.device)
        self.item_emb = nn.Embedding(num_items, self.embedding_dim).to(self.device)
        self.gat1 = GATConv(self.embedding_dim, self.embedding_dim, heads=self.heads, dropout=self.dropout, concat=True).to(self.device)
        self.gat2 = GATConv(self.embedding_dim * self.heads, self.embedding_dim, heads=1, dropout=self.dropout, concat=False).to(self.device)

        params = list(self.user_emb.parameters()) + list(self.item_emb.parameters()) + \
                 list(self.gat1.parameters()) + list(self.gat2.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=self.l2_lambda)
        criterion = nn.BCEWithLogitsLoss()

        user_edge = edge_index[0]
        item_edge = edge_index[1] - num_users

        for epoch in range(20):
            self.user_emb.train()
            self.item_emb.train()
            self.gat1.train()
            self.gat2.train()
            optimizer.zero_grad()

            x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            x = F.dropout(x, p=self.dropout, training=True)
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=True)
            x = self.gat2(x, edge_index)
            user_emb_out, item_emb_out = x[:num_users], x[num_users:]

            pos_pred = (user_emb_out[user_edge] * item_emb_out[item_edge]).sum(dim=1)
            pos_label = torch.ones_like(pos_pred)

            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_users + num_items,
                num_neg_samples=edge_index.shape[1],
                method='sparse'
            )
            neg_user = neg_edge_index[0]
            neg_item = neg_edge_index[1]

            mask = (neg_user < num_users) & (neg_item >= num_users) & (neg_item < num_users + num_items)
            neg_user = neg_user[mask]
            neg_item = neg_item[mask] - num_users

            neg_pred = (user_emb_out[neg_user] * item_emb_out[neg_item]).sum(dim=1)
            neg_label = torch.zeros_like(neg_pred)


            loss = criterion(pos_pred, pos_label) + criterion(neg_pred, neg_label)
            loss += 1e-6 * (user_emb_out.norm(2).pow(2) + item_emb_out.norm(2).pow(2))

            loss.backward()
            optimizer.step()

        self.user_emb_out = user_emb_out.detach()
        self.item_emb_out = item_emb_out.detach()

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        users_pd = users.toPandas()
        items_pd = items.toPandas()
        user_ids = users_pd['user_idx'].map(self.user2id).dropna().astype(int)
        item_ids = items_pd['item_idx'].map(self.item2id).dropna().astype(int)

        scores = torch.matmul(self.user_emb_out[user_ids], self.item_emb_out[item_ids].T)
        scores = scores.cpu().numpy()

        recs = []
        for i, user_idx in enumerate(users_pd['user_idx']):
            user_scores = scores[i]
            item_indices = np.argsort(-user_scores)[:k]
            for idx in item_indices:
                recs.append((user_idx, items_pd['item_idx'].iloc[idx], float(user_scores[idx])))

        recs_df = pd.DataFrame(recs, columns=['user_idx', 'item_idx', 'relevance'])

        if filter_seen_items and log is not None:
            log_pd = log.toPandas()
            seen = set(zip(log_pd['user_idx'], log_pd['item_idx']))
            recs_df = recs_df[~recs_df.set_index(['user_idx', 'item_idx']).index.isin(seen)]

        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(recs_df)


class GCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.convs = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) 
                                  for _ in range(num_layers)])
        self.dropout = dropout
        
    def forward(self, graph, edge_index=None):
        user_emb = self.user_emb(graph['user'].node_id)
        item_emb = self.item_emb(graph['item'].node_id)
        x = torch.cat([user_emb, item_emb])
        
        edge_index = graph['user', 'interacts', 'item'].edge_index if edge_index is None else edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        user_emb, item_emb = x.split([user_emb.size(0), item_emb.size(0)])
        return torch.matmul(user_emb, item_emb.T)
