# CS145 Recommender Systems – Team Coolios 🧠📈

This repository contains the full implementation of our models for the CS145 Recommender Systems Competition. We explored content-based, sequence-based, and graph-based recommendation paradigms to maximize **discounted revenue**.

---

## Environment Setup

Use the following commands to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/FrancoTSolis/CS145-RecSys.git
cd CS145-RecSys

# Install Java (required by Spark)
# On Linux Systems
apt-get install openjdk-17-jdk -y
# On non-Linux Systems
Manually install OpenJDK 17 and add to system path variable https://adoptium.net/temurin/releases/?version=17

# Check Java Installation
java -version

# Windows Only
Download hadoop.dll and winutils.exe from https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin

# Install package manager and dependencies
pip install -q uv
uv pip install --upgrade "lightgbm<4.0"
uv pip install torch-geometric


## Running the Models
Each model is implemented in its own .py file under the appropriate checkpoint folder. 
Make sure the model class used in the pipeline is updated in recommender_analysis_visualization.py

#include it as follows in the recommender_analysis_visualization.py script
from checkpoint2.transformer import TransformerRecommender

# then run script
!uv run recommender_analysis_visualization.py

#Repo Structure
REC_SYS/
├── checkpoint1/
│   ├── cb_hyperparams.json        # GradientBoost and Logistic Regression config
│   ├── gradBoost.py               # GradientBoostRecommender (LightGBM)
│   └── LogisticRegression.py      # LogisticRegressionRecommender
│
├── checkpoint2/
│   ├── AR.py                      # AutoRegressive Recommender
│   ├── transformer.py             # SASRec-style Transformer
│   ├── randomForest_GRU.py        # GRU and Random Forest model file
│   └── seq_config.json            # Sequence model hyperparameters
│
├── checkpoint3/
│   ├── gcn_gat.py                 # GCN + GAT model implementations
│   ├── lightGCN.py                # LightGCNRecommender
│   └── gnn_config.json            # GCN-related hyperparameters
│
├── experiments/                   # Optional experimentation scripts
├── final/                         # Final integrated strategy (optional)
├── submission.py                  # Submission runner for leaderboard
├── requirements.txt               # Package dependencies
└── README.md                      # This file


