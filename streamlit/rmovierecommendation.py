import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import requests

# API Key for TMDb (replace with your API key)
API_KEY = "ef1a72dddafa0cca7da1776e015703ed"

def get_movie_data(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    response = requests.get(url)
    data = response.json()
    return data.get("results", [])  # Return empty list if "results" key is not found

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