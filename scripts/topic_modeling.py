import pandas as pd
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download stopwords if not available
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("data/sentiment_results.csv")  # Ensure sentiment analysis saved this file

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return tokens

df['processed_text'] = df['text'].astype(str).apply(preprocess_text)

# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(df['processed_text'])
corpus = [dictionary.doc2bow(text) for text in df['processed_text']]

# Train LDA Model
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Display Topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Save model and dictionary
lda_model.save("models/lda_model")
dictionary.save("models/lda_dictionary")
