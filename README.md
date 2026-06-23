# Movie Recommendation System

> Hybrid recommender using Content-Based Filtering and Collaborative Filtering.

## Approach
- Content-Based: TF-IDF on genres + Cosine Similarity
- Collaborative: User-User similarity matrix

## Dataset
[MovieLens Dataset — Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

## Tech Stack
- Python, Pandas, Scikit-learn, NumPy

##  Quick Start
```bash
git clone git@github.com:Mohdasham/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
python recommender.py
```

##  Results
- Content-based recommendations based on genre similarity
- Collaborative filtering based on user rating patterns

