import streamlit as st
import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Clean data
movies = movies.dropna()
movies = movies.drop_duplicates()

# Convert genres
movies["genre"] = movies["genre"].str.lower()

# Vectorization
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies["genre"]).toarray()

similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    titles = movies["title"].str.lower()

    if movie not in titles.values:
        return []

    index = titles[titles == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in distances]

# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select Movie", movies["title"].values)

if st.button("Recommend"):

    result = recommend(selected_movie)

    # FINAL SAFE OUTPUT (NO ERROR POSSIBLE)
    if len(result) > 0:
        st.write("### Top 5 Recommended Movies:")
        for i in range(len(result)):
            st.write(f"{i+1}. {result[i]}")
    else:
        st.warning("No recommendations found")
