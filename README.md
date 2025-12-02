# Overview
Demo for a GitHub repo for UF DCP4300 Project. We created this project to help users find movie recommendations that are tailored to them. This eliminates spending hours scrolling to find a movie to watch.

# Authors
Chase Thompson and Mclean Dries

# This script:
1. Loads and preprocesses MovieLens 25M data
2. Builds simple baseline models
3. Trains a LightGBM model
4. Evaluates RMSE, Precision@10, and Hit Rate

# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import lightgbm as lgb
from math import sqrt

# Load MovieLens Data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Preprocessing
## Split genres into lists
movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

## One-hot encode genres
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(movies["genres"]),columns=mlb.classes_,index=movies.index)
movies = pd.concat([movies, genre_dummies], axis=1)

## Remove movies with < 20 ratings
movie_counts = ratings.groupby("movieId")["rating"].count()
valid_movies = movie_counts[movie_counts >= 20].index
ratings = ratings[ratings["movieId"].isin(valid_movies)]

## User-level features
ratings["user_avg_rating"] = ratings.groupby("userId")["rating"].transform("mean")
ratings["user_rating_count"] = ratings.groupby("userId")["rating"].transform("count")

## Movie-level features
ratings = ratings.merge(
    movies[["movieId", "title", "genres"] + list(genre_dummies.columns)],on="movieId",how="left")
ratings["movie_avg_rating"] = ratings.groupby("movieId")["rating"].transform("mean")
ratings["movie_rating_count"] = ratings.groupby("movieId")["rating"].transform("count")

## Chronological split
ratings = ratings.sort_values("timestamp")
train, test = train_test_split(ratings, test_size=0.2, shuffle=False)

# Baseline Models
## Global mean baseline
global_mean = train["rating"].mean()
global_rmse = sqrt(mean_squared_error(test["rating"], np.full(len(test), global_mean)))

## User mean baseline
user_means = train.groupby("userId")["rating"].mean()
user_pred = test["userId"].map(user_means).fillna(global_mean)
user_rmse = sqrt(mean_squared_error(test["rating"], user_pred))

## Item mean baseline
item_means = train.groupby("movieId")["rating"].mean()
item_pred = test["movieId"].map(item_means).fillna(global_mean)
item_rmse = sqrt(mean_squared_error(test["rating"], item_pred))

print("=== BASELINE RESULTS ===")
print(f"Global Mean RMSE: {global_rmse:.3f}")
print(f"User Mean RMSE:   {user_rmse:.3f}")
print(f"Item Mean RMSE:   {item_rmse:.3f}")

# LightGBM Model

# Choose features
genre_cols = list(genre_dummies.columns)
feature_cols = [
    "user_avg_rating",
    "user_rating_count",
    "movie_avg_rating",
    "movie_rating_count"
] + genre_cols

X_train = train[feature_cols]
y_train = train["rating"]

X_test = test[feature_cols]
y_test = test["rating"]

# LightGBM training
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05
    }

train_data = lgb.Dataset(X_train, y_train)
model = lgb.train(params, train_data, num_boost_round=200)

# Predictions
y_pred = model.predict(X_test)
lgbm_rmse = sqrt(mean_squared_error(y_test, y_pred))

print("\n=== LIGHTGBM RESULT ===")
print(f"LightGBM RMSE: {lgbm_rmse:.3f}")

# Precision@10 and Hit Rate
def evaluate_recommendations(test_df, preds):
    """Calculates Precision@10 and Hit Rate."""
    test_df = test_df.copy()
    test_df["pred"] = preds

    precisions = []
    hits = []

    for user in test_df["userId"].unique():
        user_data = test_df[test_df["userId"] == user]

        # Top 10 predicted movies
        top10 = user_data.sort_values("pred", ascending=False).head(10)

        # Count how many were rated >= 4
        relevant = sum(top10["rating"] >= 4)
        precisions.append(relevant / 10)

        hits.append(1 if relevant > 0 else 0)

    return np.mean(precisions), np.mean(hits)

precision10, hitrate = evaluate_recommendations(test, y_pred)

print("\n=== RECOMMENDATION QUALITY ===")
print(f"Precision@10: {precision10:.3f}")
print(f"Hit Rate:     {hitrate:.3f}")
