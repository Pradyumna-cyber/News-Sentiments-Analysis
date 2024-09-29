import requests
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Import for 3D charts
import altair as alt
import nltk
import numpy as np
import streamlit as st

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Streamlit app setup
st.title("News Insights")

# Sidebar for category selection
st.sidebar.header("News Category")
categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
selected_category = st.sidebar.selectbox("Choose a news category", categories)

# API setup
api_key = 'ea01367e3da24cc5b9532cf8c5bfc0f7'
url = 'https://newsapi.org/v2/top-headlines'

params = {
    'country': 'us',
    'category': selected_category,
    'apiKey': api_key
}

# Fetch data from API
with st.spinner("Fetching articles..."):
    response = requests.get(url, params=params)
    data = response.json()

if data.get('status') == 'ok':
    total_results = data.get('totalResults', 0)
    st.write(f"Showing {total_results} {selected_category.capitalize()} News Articles")

    articles = data.get('articles', [])
    all_descriptions = ''
    positive_text = ''
    negative_text = ''
    neutral_text = ''
    cp = 0  # Positive count
    cne = 0  # Negative count
    cnu = 0  # Neutral count
    polarities = []
    subjectivities = []
    article_polarities = []  # To store polarity of each article

    # Function to create and display a static word cloud
    def create_wordcloud(text):
        if text.strip():  # Ensure text is not empty
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            return fig
        return None

    # Sentiment Overview at the top
    st.write("## Sentiment Overview")

    # Display news articles as cards
    search_query = st.sidebar.text_input("Search Articles", "")
    filtered_articles = [article for article in articles if search_query.lower() in article['title'].lower()]

    if filtered_articles:
        for idx, article in enumerate(filtered_articles, start=1):
            # Extract article details
            title = article.get('title', 'No title available')
            desp = article.get('description', 'No description available')
            url = article.get('url', '#')
            image_url = article.get('urlToImage', None)

            # Initialize sentiment for each article
            sentiment = 'Neutral'
            article_polarity = 0  # Average polarity per article

            # Process sentiment analysis
            if desp and desp != 'No description available':
                all_descriptions += desp + ' '
                sentences = sent_tokenize(desp)
                if sentences:
                    for sentence in sentences:
                        words = word_tokenize(sentence)
                        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                        lemmatized_sentence = ' '.join(lemmatized_words)

                        blob = TextBlob(lemmatized_sentence)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity
                        polarities.append(polarity)
                        subjectivities.append(subjectivity)

                        article_polarity += polarity

                    # Average polarity for the article
                    article_polarity /= len(sentences)

                    # Determine overall sentiment for the article
                    if article_polarity > 0:
                        sentiment = 'Positive'
                        cp += 1
                        positive_text += desp + ' '
                    elif article_polarity < 0:
                        sentiment = 'Negative'
                        cne += 1
                        negative_text += desp + ' '
                    else:
                        sentiment = 'Neutral'
                        cnu += 1
                        neutral_text += desp + ' '
                else:
                    sentiment = 'Neutral'
                    article_polarities.append(0)
            else:
                # If no description, maintain neutral sentiment
                article_polarities.append(0)

            # Color-coded sentiment display
            sentiment_color = {
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }

            # Create a card for each article
            with st.container():
                st.write(f"### {title}")
                if image_url:
                    st.image(image_url, width=400)
                st.write(desp)
                st.markdown(f"[Read more...]({url})")

                # Display the sentiment for each article along with polarity score
                st.write(f"**Sentiment:** <span style='color:{sentiment_color[sentiment]}'>{sentiment}</span> | **Polarity Score:** {article_polarity:.2f}", unsafe_allow_html=True)
                st.write("---")

        # Show the static word cloud for all descriptions
        st.sidebar.header("Static Word Cloud")
        if all_descriptions.strip():  # Ensure non-empty text
            wordcloud_fig = create_wordcloud(all_descriptions)
            if wordcloud_fig:
                st.sidebar.pyplot(wordcloud_fig)
            else:
                st.sidebar.write("Not enough text for word cloud.")
        else:
            st.sidebar.write("No descriptions available to generate a word cloud.")

    else:
        st.error("No articles found.")
else:
    st.error(f"Error: {data.get('message')} (Status code: {response.status_code})")

# Sentiment distribution graphs
st.sidebar.subheader("Sentiment Distribution")

# Create a DataFrame for sentiment counts
d = {'Sentiments': ['Positive', 'Negative', 'Neutral'], 'Count': [cp, cne, cnu]}
df = pd.DataFrame(d)
# Toggle between 3D Pie Chart and 3D Bar Chart
chart_type = st.sidebar.radio("Choose Chart Type", ['Pie Chart', 'Bar Chart'])

if chart_type == 'Pie Chart':
    # Display a Plotly 3D pie chart for sentiment distribution
    fig_pie = go.Figure(data=[go.Pie(labels=df['Sentiments'], values=df['Count'], hole=.3, textinfo='label+value')])
    fig_pie.update_layout(title_text='Sentiment Distribution', title_x=0.5)
    st.sidebar.plotly_chart(fig_pie)

else:
    # Display a Plotly 3D bar chart for sentiment distribution with value labels
    fig_bar = go.Figure(data=[go.Bar(x=df['Sentiments'], y=df['Count'], text=df['Count'], textposition='auto',
                                     marker_color=['#66b3ff', '#ff6666', '#99ff99'])])
    fig_bar.update_layout(title_text='Sentiment Distribution', title_x=0.5, scene=dict(zaxis_title='Count'))
    st.sidebar.plotly_chart(fig_bar)
