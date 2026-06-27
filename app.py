!pip install pandas
!pip install scikit-learn

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\DELL\Downloads\movies.csv")

df = df.head(250)

df = df.rename(columns={"id":"movie_id"})

genre = [
    "Action","Comedy","Drama","Romance",
    "mystery","crime","family" "biography"
]
movies_df = df[["movie_id","title","genre","ratings"]]

movies_df.to_csv("movies_250.csv", index=False)

print(movies_df.head())
print("Total Movies:", len(movies_df))

import pandas as pd

movies = pd.read_csv(r"C:\Users\DELL\Downloads\movies.csv")

print(movies.head())
print("\nTotal Movies:", len(movies))

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

genre_matrix = cv.fit_transform(movies["genre"])

print(genre_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(genre_matrix)

print(similarity.shape)

movies = movies.drop_duplicates()

movies["genre"] = movies["genre"].fillna("")

print("Missing Values:")
print(movies.isnull().sum())

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

genre_matrix = cv.fit_transform(
    movies["genre"]
)

print("Genre Matrix Shape:")
print(genre_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(
    genre_matrix
)

print("Similarity Matrix Created")
print(similarity.shape)

def recommend(movie_name):

    movie_name = movie_name.lower()

    matches = movies[
        movies["title"].str.lower().str.contains(
            movie_name,
            na=False
        )
    ]

    if len(matches) == 0:
        return ["Movie Not Found"]

    movie_index = matches.index[0]

    distances = list(
        enumerate(similarity[movie_index])
    )

    movie_list = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for movie in movie_list:

        recommendations.append(
            movies.iloc[movie[0]].title
        )

    return recommendations

for movie in df["title"].head(10):
    print(movie)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv(r"C:\Users\DELL\Downloads\movies.csv")

movies = movies.drop_duplicates()

movies["genre"] = movies["genre"].fillna("")

cv = CountVectorizer()

genre_matrix = cv.fit_transform(
    movies["genre"]
)

similarity = cosine_similarity(
    genre_matrix
)
def recommend(movie_name):

    movie_name = movie_name.lower()

    matches = movies[
        movies["title"].str.lower().str.contains(
            movie_name,
            na=False
        )
    ]

    if len(matches) == 0:
        return ["Movie Not Found"]

    movie_index = matches.index[0]

    distances = list(
        enumerate(similarity[movie_index])
    )

    movie_list = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []
for movie in movie_list:
        recommendations.append(
            movies.iloc[movie[0]].title
        )

    return recommendations

print("Movie Recommendation System")

movie_name = input(
    "\nEnter Movie Name: "
)

result = recommend(movie_name)

print("\nTop 5 Recommended Movies:\n")

for movie in result:
    print(movie)