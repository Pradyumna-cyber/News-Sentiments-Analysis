import requests
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk


# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# API setup
api_key = 'ea01367e3da24cc5b9532cf8c5bfc0f7'
url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=ea01367e3da24cc5b9532cf8c5bfc0f7'

params = {
    'country': 'us',
    'apiKey': api_key
}

response = requests.get(url, params=params)
data = response.json()

if data.get('status') == 'ok':
    total_results = data.get('totalResults', 0)
    print(f"Total results found: {total_results}")

    articles = data.get('articles', [])
    all_descriptions = ''
    positive_text = ''
    negative_text = ''
    neutral_text = ''
    cp = 0
    cne = 0
    cnu = 0
    polarities = []
    subjectivities = []
    article_polarities = []  # To store polarity of each article

    if articles:
        for idx, article in enumerate(articles, start=1):
            desp = article.get('description', 'No description available')
            if desp == 'No description available':
                continue
            print(f"{idx}. {desp}")

            if desp and desp != 'No description available':
                all_descriptions += desp + ' '
                sentences = sent_tokenize(desp)
                if sentences:
                    article_polarity = 0
                    for sentence in sentences:
                        words = word_tokenize(sentence)
                        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                        lemmatized_sentence = ' '.join(lemmatized_words)

                        print(f"Original: {sentence}")
                        print(f"Lemmatized: {lemmatized_sentence}")

                        blob = TextBlob(lemmatized_sentence)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity
                        polarities.append(polarity)
                        subjectivities.append(subjectivity)

                        article_polarity += polarity  # Sum polarity of each sentence

                        if polarity > 0:
                            sentiment = 'Positive'
                            cp += 1
                            positive_text += lemmatized_sentence + ' '
                        elif polarity < 0:
                            sentiment = 'Negative'
                            cne += 1
                            negative_text += lemmatized_sentence + ' '
                        else:
                            sentiment = 'Neutral'
                            cnu += 1
                            neutral_text += lemmatized_sentence + ' '
                        print(f"    Sentiment: {sentiment}")

                    # Average polarity for the article
                    article_polarity /= len(sentences)
                    article_polarities.append(article_polarity)
                    print(f"    Article Polarity: {article_polarity}\n")

        # Generate and display word cloud for all descriptions
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for All Descriptions')
        plt.show()

        # Generate word clouds for positive, negative, and neutral sentiments
        def generate_wordcloud(text, title):
            if text:  # Check if text is non-empty
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(title)
                plt.show()

        generate_wordcloud(positive_text, 'Positive Sentiment Word Cloud')
        generate_wordcloud(negative_text, 'Negative Sentiment Word Cloud')
        generate_wordcloud(neutral_text, 'Neutral Sentiment Word Cloud')

    else:
        print("No articles found.")
else:
    print("Error:", data.get('message'))
    print("Status code:", response.status_code)

# Create and display sentiment distribution pie chart
d = {'Sentiments': ['Positive', 'Negative', 'Neutral'], 'Count': [cp, cne, cnu]}
df = pd.DataFrame(d)
print(df)

# Create and display sentiment distribution bar plot
palette_color = sns.color_palette('bright')
plt.figure(figsize=(8, 6))
sns.barplot(x='Sentiments', y='Count', data=df, palette=palette_color)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiments')
plt.ylabel('Count')
plt.show()
