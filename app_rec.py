from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()

# Load the data
ratings = pd.read_csv(r'C:\Users\rafael.mj\Desktop\FastAPI\ratings.csv', sep=',')
movies = pd.read_csv(r'C:\Users\rafael.mj\Desktop\FastAPI\movies.csv', sep=',')

# Configure Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Precompute the user-movie matrix and similarity matrices
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
movie_similarity = cosine_similarity(user_movie_matrix.T)  # Movie-movie similarity
user_similarity = cosine_similarity(user_movie_matrix)     # User-user similarity
movie_ids = user_movie_matrix.columns                      # List of movieIds
user_ids = user_movie_matrix.index                         # List of userIds

# 1. Best Seller Function
def best_seller_recommendations_ratings(ratings, movies, top_n=10):
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    min_ratings = 10
    movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]

    best_sellers = movie_stats.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False])
    best_sellers = best_sellers.merge(movies, on='movieId')

    return best_sellers.head(top_n).to_dict(orient='records')

# 2. Most Seen Function
def most_seen(ratings, movies, top_n=10):
    movie_views = ratings.groupby('movieId').size().reset_index(name='num_views')
    min_views = 10
    movie_views = movie_views[movie_views['num_views'] >= min_views]
    most_viewed = movie_views.sort_values(by='num_views', ascending=False)
    most_viewed = most_viewed.merge(movies, on='movieId')
    return most_viewed.head(top_n).to_dict(orient='records')

# 3. Similar Movies Function
def similarity_1(movie_id: int, movies, movie_ids, movie_similarity, top_n=10):
    if movie_id not in movie_ids:
        return {'error': f'Movie ID {movie_id} not found in ratings data'}
    
    movie_idx = np.where(movie_ids == movie_id)[0][0]
    similarity_scores = movie_similarity[movie_idx]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    similar_movies_ids = movie_ids[similar_indices]
    
    similar_movies = movies[movies['movieId'].isin(similar_movies_ids)].copy()
    similar_movies['similarity'] = [similarity_scores[idx] for idx in similar_indices]
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)
    
    return similar_movies.to_dict(orient='records')

# 4. User-Based Recommendations Function
def user_based_recommendations(user_id: int, ratings, movies, user_movie_matrix, user_similarity, user_ids, movie_ids, top_n=10, k=5):
    if user_id not in user_ids:
        return {'error': f'User ID {user_id} not found in ratings data'}
    
    user_idx = np.where(user_ids == user_id)[0][0]
    similarity_scores = user_similarity[user_idx]
    similar_user_indices = similarity_scores.argsort()[::-1][1:k + 1]
    similar_users = user_ids[similar_user_indices]
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    similar_users_ratings = ratings[ratings['userId'].isin(similar_users)]
    candidate_movies = similar_users_ratings[~similar_users_ratings['movieId'].isin(user_rated_movies)]

    movie_scores = candidate_movies.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    weights = similarity_scores[similar_user_indices]
    movie_scores['weighted_score'] = movie_scores['avg_rating'] * (weights.sum() / k)

    recommendations = movie_scores.sort_values(by='weighted_score', ascending=False)
    recommendations = recommendations.merge(movies, on='movieId')
    
    return recommendations.head(top_n).to_dict(orient='records')

# Endpoint for the root page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FastAPI Endpoints
@app.get("/best-seller/ratings/")
async def get_best_seller_ratings(top_n: int = 10):
    return best_seller_recommendations_ratings(ratings, movies, top_n)

@app.get('/best-seller/views/')
def get_most_seen(top_n: int = 10):
    return most_seen(ratings, movies, top_n)

@app.get("/similarity/{movie_id}")
def get_similarity(movie_id: int, top_n: int = 10):
    return similarity_1(movie_id, movies, movie_ids, movie_similarity, top_n)

@app.get("/user-based/{user_id}")
def get_user_recommendations(user_id: int, top_n: int = 10, k: int = 5):
    return user_based_recommendations(user_id, ratings, movies, user_movie_matrix, user_similarity, user_ids, movie_ids, top_n, k)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_rec:app", host="127.0.0.1", port=8000, reload=True)