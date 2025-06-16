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





class LogisticRegressionRecommender(BaseRecommender):
    def __init__(self, seed=None, penalty='l2', C=1.0, spark_session=None):
        super().__init__(seed)
        solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            random_state=self.seed,
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.spark_session = spark_session

    def fit(self, log: DataFrame, user_features=None, item_features=None, cv=5):
        if user_features and item_features:
            self.spark_session = log.sql_ctx.sparkSession

            joined = log.join(user_features, on='user_idx').join(item_features, on='item_idx')
            pd_data = joined.drop('user_idx', 'item_idx', '__iter').toPandas()

            pd_data = pd.get_dummies(pd_data)

            if 'price' in pd_data.columns:
                pd_data['price'] = self.scaler.fit_transform(pd_data[['price']])

            y = pd_data['relevance']
            X = pd_data.drop(['relevance'], axis=1)

            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.seed)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model_cv = LogisticRegression(
                    penalty=self.model.penalty,
                    C=self.model.C,
                    solver=self.model.solver,
                    random_state=self.seed,
                    max_iter=1000
                )
                model_cv.fit(X_train, y_train)
                y_pred = model_cv.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                scores.append(acc)

            print(f"[CV] Accuracy scores: {scores}")
            print(f"[CV] Mean accuracy: {np.mean(scores):.4f}")

            
            # --- Final Model Fit ---
            self.model.fit(X, y)
            self.feature_columns = X.columns

    def pandas_to_spark(df, spark_session):
        return spark_session.createDataFrame(df)

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        cross = users.join(items).drop('__iter').toPandas().copy()

        cross['orig_price'] = cross['price']

        cross = pd.get_dummies(cross)

        cross['price'] = self.scaler.transform(cross[['price']])

        cross['prob'] = self.model.predict_proba(
            cross.drop(['user_idx', 'item_idx', 'orig_price'], axis=1)
        )[:, np.where(self.model.classes_ == 1)[0][0]]

        cross['relevance'] = (np.sin(cross['prob']) + 1) * np.exp(cross['prob'] - 1) * \
                            np.log1p(cross["orig_price"]) * np.cos(cross["orig_price"] / 100) * \
                            (1 + np.tan(cross['prob'] * np.pi / 4))

        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)
        cross['price'] = cross['orig_price']

        return self.pandas_to_spark(cross[['user_idx', 'item_idx', 'relevance', 'price']], spark_session=self.spark_session)