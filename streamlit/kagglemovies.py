import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Function to load movie data from CSV file
def get_movie_data(csv_file):
    # Load data from CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract relevant information
    movie_data = []
    for index, row in df.iterrows():
        movie_info = {
            "title": row.get('Series_Title', ''),
            "overview": row.get('Overview', ''),
            "release_date": row.get('Released_Year', ''),
            "poster_link": row.get('Poster_Link', ''),  
            "genre": row.get('Genre', ''),  
            "imdb_rating": row.get('IMDB_Rating', ''),  
            "director": row.get('Director', '')
        }
        movie_data.append(movie_info)

    return movie_data

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to preprocess text using BERT tokenizer
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    return inputs

# Function to encode text using BERT model
def encode_text(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs[0].mean(dim=1)

# Function to recommend movies based on user input
def recommend_movies(input_text, csv_file):
    movie_data = get_movie_data(csv_file)
    filtered_movies = []
    for movie in movie_data:
        if input_text.lower() in movie["title"].lower() or \
           input_text.lower() in movie["director"].lower() or \
           input_text.lower() in movie["genre"].lower() or \
           input_text.lower() in movie["overview"].lower():
            filtered_movies.append(movie)
    return filtered_movies[:10]  # Return only the top 10 movies

# Main function to create the UI
def main(csv_file):
    st.title("🎬 Movie Recommendation App")

    # User input
    input_text = st.text_input("Enter a movie name, director, or cast", "")

    if st.button("Get Recommendations"):
        if input_text:
            # Get recommendations based on user input
            recommendations = recommend_movies(input_text, csv_file)
            
            # Display recommendations
            if recommendations:
                st.subheader("🌟 Recommended Movies:")
                for movie in recommendations:
                    st.markdown("---")
                    st.image(movie['poster_link'], caption=movie['title'], width=200)
                    st.write(f"**Title:** {movie['title']}")
                    st.write(f"**Overview:** {movie['overview']}")
                    st.write(f"**Release Date:** {movie['release_date']}")
                    st.write(f"**Genre:** {movie['genre']}")
                    st.write(f"**IMDB Rating:** {movie['imdb_rating']}")
                    st.write(f"**Director:** {movie['director']}")
                    st.markdown("---")
            else:
                st.error("No recommendations found. Please try another movie.")

if __name__ == "__main__":
    csv_file = 'kaggle/imdb_top_1000.csv'  # Update with the correct file path
    main(csv_file)