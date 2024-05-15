# import streamlit as st
# from transformers import pipeline

# def main():
#     st.title("Sentiment Analysis App")
#     st.markdown("---")

#     # Load the sentiment analysis model
#     sentiment_model = pipeline("sentiment-analysis")

#     # Add a text input widget for user input
#     text = st.text_area("Enter text for sentiment analysis", "I love this product!")

#     # Add some space
#     st.write("")
    
#     # Perform sentiment analysis when the user submits the text
#     if st.button("Analyze Sentiment", key="analyze_button"):
#         with st.spinner('Analyzing...'):
#             result = sentiment_model(text)[0]
#             sentiment = result['label']
#             score = result['score']
            
#             # Display the sentiment result
#             st.success(f"Sentiment: {sentiment}")
#             st.info(f"Confidence Score: {score}")

# if __name__ == "__main__":
#     main()
import streamlit as st
from PIL import Image
from transformers import pipeline

def main():
    # Set page width and title
    st.set_page_config(layout="wide")
    st.title("Sentiment Analysis App")
    st.markdown("---")

    # Load the sentiment analysis model
    sentiment_model = pipeline("sentiment-analysis")

    # Add header image
    image = Image.open("sentiment_analysis.jpg")
    st.image(image, use_column_width=True)

    # Add a text input widget for user input
    text = st.text_area("Enter text for sentiment analysis", "I love this product!")

    # Add a slider for adjusting the confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Add space for better separation
    st.write("")
    
    # Perform sentiment analysis when the user submits the text
    if st.button("Analyze Sentiment", key="analyze_button"):
        with st.spinner('Analyzing...'):
            result = sentiment_model(text)[0]
            sentiment = result['label']
            score = result['score']

            # Display the sentiment result with colored badges
            if score >= confidence_threshold:
                st.success(f"Sentiment: {sentiment}")
                st.info(f"Confidence Score: {score}")
            else:
                st.warning("Low confidence score. Try another text.")

if __name__ == "__main__":
    main()