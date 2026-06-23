import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Sample movie dataset
movies = pd.DataFrame({
    'movie_id': range(1, 21),
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
        'Avengers: Endgame', 'Iron Man', 'Captain America', 'Thor',
        'Titanic', 'La La Land', 'The Notebook', 'Forrest Gump',
        'Toy Story', 'Finding Nemo', 'The Lion King', 'Shrek',
        'Parasite', 'The Godfather', 'Pulp Fiction', 'Fight Club'
    ],
    'genre': [
        'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi', 'Action Sci-Fi',
        'Action Adventure Fantasy', 'Action Sci-Fi', 'Action Adventure', 'Action Adventure Fantasy',
        'Drama Romance', 'Drama Music Romance', 'Drama Romance', 'Drama Romance',
        'Animation Adventure Comedy', 'Animation Adventure Comedy', 'Animation Drama', 'Animation Comedy Fantasy',
        'Drama Thriller', 'Crime Drama', 'Crime Drama Thriller', 'Drama Thriller'
    ],
    'rating': [9.0, 8.8, 8.6, 8.7, 8.4, 7.9, 7.8, 7.9, 7.8, 8.0, 7.9, 8.8, 8.3, 8.1, 8.5, 7.9, 8.5, 9.2, 8.9, 8.8]
})

ratings = pd.DataFrame({
    'user_id': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
    'movie_id': [1,2,3,4,1,3,5,6,2,4,7,8,9,10,11,12,13,14,15,16],
    'rating': [5,5,4,5,3,4,5,4,5,4,3,4,4,5,5,3,5,4,5,4]
})

print("🎬 Movie Recommendation System")
print(f"Movies: {len(movies)}, Ratings: {len(ratings)}")

#Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_based_recommend(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genre', 'rating']].iloc[movie_indices]

#Collaborative Filtering
user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def collab_recommend(user_id, n=5):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:4].index
    similar_user_ratings = user_movie_matrix.loc[similar_users].mean(axis=0)
    user_rated = user_movie_matrix.loc[user_id]
    unrated_movies = user_rated[user_rated == 0].index
    recommendations = similar_user_ratings[unrated_movies].sort_values(ascending=False).head(n)
    return movies[movies['movie_id'].isin(recommendations.index)][['title', 'genre', 'rating']]

#Demo
print("\n Content-Based (similar to 'Inception'):")
print(content_based_recommend('Inception'))

print("\nCollaborative Filtering (for User 1):")
print(collab_recommend(1))