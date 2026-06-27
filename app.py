import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pd.read_csv("movies.csv")

# Keep required columns only
movies = movies[["movie_id", "title", "genre","ratings"]]

# Clean data
movies.dropna(inplace=True)
movies.drop_duplicates(subset="title", inplace=True)

# Convert genres to lowercase
movies["genre"] = movies["genre"].str.lower()

# -----------------------------
# MODEL BUILDING
# -----------------------------
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies["genre"]).toarray()

similarity = cosine_similarity(vectors)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie_title):

    movie_title = movie_title.lower()

    movies_list = movies["title"].str.lower()

    if movie_title not in movies_list.values:
        return []

    index = movies_list[movies_list == movie_title].index[0]

    distances = list(enumerate(similarity[index]))

    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    return [movies.iloc[i[0]].title for i in distances]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Get Top 5 movies based on genre similarity")

selected_movie = st.selectbox("Choose a movie", movies["title"].values)

if st.button("Recommend"):

    results = recommend(selected_movie)

    if results:
        st.success("Top 5 Recommended Movies:")

        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Movie not found in dataset")