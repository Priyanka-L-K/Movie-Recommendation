import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import requests
from imdb import IMDb

# Create an instance of the IMDb class
ia = IMDb()

def get_movie_data(query):
    # Search for movies using the query
    movies = ia.search_movie(query)

    # Extract relevant information including poster links
    movie_data = []
    for movie in movies:
        ia.update(movie)
        movie_info = {
            "title": movie['title'],
            "overview": movie.get('plot outline', ''),
            "release_date": movie.get('year', ''),
            "poster_url": movie.get('full-size cover url', ''),
            "genres": ', '.join(movie.get('genres', [])),
            "rating": movie.get('rating', ''),
            "directors": ', '.join([director['name'] for director in movie.get('directors', [])]),
            "writers": ', '.join([writer['name'] for writer in movie.get('writers', [])]),
            "cast": ', '.join([actor['name'] for actor in movie.get('cast', [])[:5]])  # Limit to first 5 actors
        }
        movie_data.append(movie_info)

    return movie_data

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    return inputs

def encode_text(text):
    # Encode input text using BERT
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs[0].mean(dim=1)

def recommend_movies(input_text):
    # Encode user input text
    input_embedding = encode_text(input_text)

    # Get movie data based on user input
    movie_data = get_movie_data(input_text)

    # Extract relevant information for recommendations
    recommendations = []
    for movie in movie_data:
        recommendations.append({
            "title": movie["title"],
            "overview": movie["overview"],
            "release_date": movie["release_date"]
        })

    return recommendations

def main():
    st.title("Movie Recommendation App")

    # User input
    input_text = st.text_input("Enter a movie name, director, or cast", "")

    if st.button("Get Recommendations"):
        if input_text:
            # Get recommendations based on user input
            recommendations = recommend_movies(input_text)
            
            # Display recommendations
            if recommendations:
                st.write("Recommended Movies:")
                for movie in recommendations:
                    st.write(f"Title: {movie['title']}")
                    st.write(f"Overview: {movie['overview']}")
                    st.write(f"Release Date: {movie['release_date']}")
                    st.write("----")
            else:
                st.write("No recommendations found. Please try another movie.")

if __name__ == "__main__":
    main()