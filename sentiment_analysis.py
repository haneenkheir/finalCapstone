
# Import libraries and load spaCy model 
import spacy 
from textblob import TextBlob
import pandas as pd 
from itertools import combinations
nlp = spacy.load("en_core_web_sm")

# Load dataset 
df = pd.read_csv('amazon_product_reviews.csv')

# Initial data cleaning
# Selecting relevant columns and removing missing values
df = df[['reviews.text']].dropna()
df['reviews.text'] = df['reviews.text'].str.lower()

# Sentiment analysis 
# Function to analyse sentiment of a product review 

def analyse_sentiment(review):
    doc = nlp(review)
    review = review.strip()
    processed_text = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

    # Analyse polarity 
    blob = TextBlob(processed_text)
    polarity_score = blob.sentiment.polarity

    # Assign sentiment based on polarity score 
    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment  

# Model evaluation 

# df sample 
df= df.sample(5000, random_state=42)

# Apply polarity analysis 

df['sentiment'] = df['reviews.text'].apply(analyse_sentiment)


# select negative and positive reviews 
negative_review1 = df[df['sentiment'] == 'negative']['reviews.text'].iloc[3]
negative_review2 = df[df['sentiment'] == 'negative']['reviews.text'].iloc[0]

positive_review1 = df[df['sentiment'] == 'positive']['reviews.text'].iloc[0]
positive_review2 = df[df['sentiment'] == 'positive']['reviews.text'].iloc[4]

# Function to calculate semantic similarity
def calculate_similarity(review1, review2):
    doc1 = nlp(review1)
    doc2 = nlp(review2)
    similarity = doc1.similarity(doc2)
    return similarity

print("Negative Review 1:", negative_review1)
print("Negative Review 2:", negative_review2)
print("Positive Review 1:", positive_review1)
print("Positive Review 2:", positive_review2)
print()


# Calculate similarity between negative reviews
negative_similarity = calculate_similarity(negative_review1, negative_review2)
print("Similarity between Negative Reviews 1 and 2:", negative_similarity)

# Calculate similarity between positive reviews
positive_similarity = calculate_similarity(positive_review1, positive_review2)
print("Similarity between Positive Reviews 1 and 2:", positive_similarity)

# Calculate similarity between a negative review and a positive review
negative_positive_similarity = calculate_similarity(negative_review1, positive_review1)
print("Similarity between Negative Review 1 and Positive Review 1:", negative_positive_similarity)







