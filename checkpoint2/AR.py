import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pyspark.sql import functions as sf
from pyspark.sql.window import Window
from pyspark.sql import DataFrame

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pyspark.sql.functions as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


from collections import defaultdict, Counter, deque
import pandas as pd
import numpy as np
import pyspark.sql.functions as sf
from pyspark.sql import Window
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def pandas_to_spark(df, spark_session):
    return spark_session.createDataFrame(df)

class ARRecommender:
    def __init__(self, order=2, max_seq_len=100, seed=None, spark=None, smoothing=None, k=0.001):
        super().__init__()
        self.order = order
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.spark = spark
        self.smoothing = smoothing
        self.k = k
        self.user_sequences = defaultdict(lambda: deque(maxlen=self.max_seq_len))
        self.ngram_counts = defaultdict(Counter)
        self.item_probs = {}
        self.item_prices = {}
        self.item_scores = defaultdict(float)

    def fit(self, log, user_features=None, item_features=None):
        if self.smoothing != None:
            available_cols = log.columns
            optional_cols = {
                "timestamp": sf.lit(0),
                "price": sf.lit(1.0)
            }

            for col, default in optional_cols.items():
                if col not in available_cols:
                    log = log.withColumn(col, default)

            log_pd = log.select("user_idx", "item_idx", "timestamp", "price", "relevance").toPandas()

            for row in log_pd.itertuples(index=False):
                user = row.user_idx
                item = row.item_idx
                timestamp = row.timestamp
                price = row.price
                response = row.relevance
                self.user_sequences[user].append((item, timestamp, response, price))
                self.item_prices[item] = price
                self.item_scores[item] += response

            for user, interactions in self.user_sequences.items():
                sequence = [item for item, _, _, _ in interactions]
                for n in range(1, self.order + 1):
                    for i in range(len(sequence) - n):
                        context = tuple(sequence[i:i+n])
                        next_item = sequence[i+n]
                        self.ngram_counts[context][next_item] += 1

            self._compute_probs()
        else:
            available_cols = log.columns
            optional_cols = {
                "timestamp": sf.lit(0),
                "price": sf.lit(1.0)
            }

            for col, default in optional_cols.items():
                if col not in available_cols:
                    log = log.withColumn(col, default)

            log_pd = log.select("user_idx", "item_idx", "timestamp", "price", "relevance").toPandas()

            for row in log_pd.itertuples(index=False):
                user = row.user_idx
                item = row.item_idx
                timestamp = row.timestamp
                price = row.price
                response = row.relevance
                self.user_sequences[user].append((item, timestamp, response, price))

            for user, interactions in self.user_sequences.items():
                for item, _, response, _ in interactions:
                    self.item_scores[item] += response

    def _compute_probs(self):
        self.item_probs = {}

        if self.smoothing == "add-k":
            for context, counter in self.ngram_counts.items():
                total = sum(counter.values()) + self.k * len(self.item_scores)
                self.item_probs[context] = {
                    item: (counter[item] + self.k) / total for item in self.item_scores
                }

        elif self.smoothing == "back-off":
            for context, counter in self.ngram_counts.items():
                total = sum(counter.values())
                self.item_probs[context] = {item: count / total for item, count in counter.items()}

    def _get_prob_backoff(self, context, item):
        for n in range(len(context), -1, -1):
            sub_context = context[-n:]
            if tuple(sub_context) in self.item_probs and item in self.item_probs[tuple(sub_context)]:
                return self.item_probs[tuple(sub_context)][item]
        return 1e-6

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        if self.smoothing != None:
            cross = users.crossJoin(items)

            if filter_seen_items and log is not None:
                seen = log.select("user_idx", "item_idx")
                cross = cross.join(seen, on=["user_idx", "item_idx"], how="left_anti")

            cross_pd = cross.toPandas()

            relevance_scores = []
            for row in cross_pd.itertuples(index=False):
                user = row.user_idx
                item = row.item_idx
                context = tuple([x[0] for x in list(self.user_sequences[user])[-self.order:]])

                if self.smoothing == "add-k":
                    prob = self.item_probs.get(context, {}).get(item, self.k / (self.k * len(self.item_scores)))
                elif self.smoothing == "back-off":
                    prob = self._get_prob_backoff(context, item)
                else:
                    prob = 1e-6

                price = self.item_prices.get(item, getattr(row, "price", 1.0))
                expected_revenue = price * prob
                relevance_scores.append(expected_revenue)

            cross_pd["relevance"] = relevance_scores
            cross_pd = cross_pd.sort_values(by=["user_idx", "relevance"], ascending=[True, False])
            topk_pd = cross_pd.groupby("user_idx").head(k)

            return pandas_to_spark(topk_pd[["user_idx", "item_idx", "relevance"]], self.spark)
        else:
            cross = users.crossJoin(items)

            if filter_seen_items and log is not None:
                seen = log.select("user_idx", "item_idx")
                cross = cross.join(seen, on=["user_idx", "item_idx"], how="left_anti")

            cross_pd = cross.toPandas()

            relevance_scores = []
            for row in cross_pd.itertuples(index=False):
                user = row.user_idx
                item = row.item_idx
                context = tuple([x[0] for x in list(self.user_sequences[user])[-self.order:]])
                prob = self.item_probs.get(context, {}).get(item, 0.0001)
                price = self.item_prices.get(item, getattr(row, "price", 1.0))
                expected_revenue = price * prob
                relevance_scores.append(expected_revenue)

            cross_pd["relevance"] = relevance_scores

            cross_pd = cross_pd.sort_values(by=["user_idx", "relevance"], ascending=[True, False])
            topk_pd = cross_pd.groupby("user_idx").head(k)

            return pandas_to_spark(topk_pd[["user_idx", "item_idx", "relevance"]], self.spark)
