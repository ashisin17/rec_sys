# CS145 Final Project — Team Coolios

## Project Overview

We implemented and evaluated content-based, sequence-based, and graph-based recommendation models to maximize **discounted revenue** using the Sim4Rec framework. Our final models include:

- `GradientBoostRecommender` (Checkpoint 1)
- `TransformerRecommender` (Checkpoint 2)
- `LightGCNRecommender` (Checkpoint 3)

Our submission combines price-aware scoring, sequence modeling, and graph-based collaborative filtering.

---

## Repository Structure

cs145-recsys-team/
├── checkpoint1/ # Content-based recommenders
│ ├── gradboost.py
│ └── cb_hyperparams.json
├── checkpoint2/ # Sequence-based recommenders
│ ├── transformer.py
│ └── seq_config.json
├── checkpoint3/ # Graph-based recommenders
│ ├── light_gcn.py
│ └── gnn_config.json
├── final/ # Final integrated analysis (plots, ablations, etc.)
├── experiments/ # Optional tuning scripts or notebooks
├── submission.py # Final leaderboard submission entry point
├── requirements.txt # Python dependencies
└── README.md # Setup and usage instructions

yaml
Copy
Edit

---

## Setup Instructions

### 1. Install dependencies

###.... completelt

👥 Team Coolios
Ashita Singh
Megan Jacob
Radhika Kakkar
Vishaka Bhat