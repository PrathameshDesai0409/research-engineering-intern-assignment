import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from textblob import TextBlob  # For sentiment analysis
import torch
import json

# Cached function to load JSONL dataset
@st.cache_data
def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                post = json.loads(line)
                # Extract relevant fields from the nested 'data' structure
                post_data = post.get('data', {})
                data.append({
                    'title': post_data.get('title', ''),
                    'selftext': post_data.get('selftext', ''),
                    'subreddit': post_data.get('subreddit', ''),
                    'author': post_data.get('author', ''),
                    'created_utc': post_data.get('created_utc', None),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0)
                })
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return pd.DataFrame(data)

# Cached function to preprocess data
@st.cache_data
def preprocess_data(df):
    try:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        df['date'] = df['created_utc'].dt.date
        df['text'] = df['title'] + " " + df['selftext'].fillna("")
        df['text_length'] = df['text'].apply(len)
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['sentiment_category'] = df['sentiment'].apply(
            lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
        )
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
    return df

# Load and preprocess data
with st.spinner('Loading and preprocessing data...'):
    df = load_jsonl("data/dataset.jsonl")
    df = preprocess_data(df)

# Topic modeling using LDA
@st.cache_data
def perform_topic_modeling(df):
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform(df['text'])
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(tfidf)
        return lda, vectorizer, tfidf
    except Exception as e:
        st.error(f"Error performing topic modeling: {e}")
        return None, None, None

lda, vectorizer, tfidf = perform_topic_modeling(df)

# Get topic names based on top words
def get_topic_names(model, vectorizer, n_top_words=5):
    topic_names = []
    try:
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_names.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
    except Exception as e:
        st.error(f"Error getting topic names: {e}")
    return topic_names

topic_names = get_topic_names(lda, vectorizer) if lda else []

# Initialize chatbot model (GPT-2 from Hugging Face)
@st.cache_resource
def load_chatbot():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        return chatbot
    except Exception as e:
        st.error(f"Error loading chatbot: {e}")
        return None

chatbot = load_chatbot()

# Streamlit app
st.title("Social Media Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", df['date'].min())
end_date = st.sidebar.date_input("End Date", df['date'].max())
keyword = st.sidebar.text_input("Search Keyword")

# Filter data
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
if keyword:
    filtered_df = filtered_df[filtered_df['text'].str.contains(keyword, case=False)]

# Visualizations
st.header("Visualizations")

# Time series of posts
st.subheader("Time Series of Posts")
time_series = filtered_df.set_index('created_utc').resample('D').size().reset_index(name='count')
fig1 = px.line(time_series, x='created_utc', y='count', title='Number of Posts Over Time')
st.plotly_chart(fig1)

# Topic distribution
if lda and tfidf is not None:
    st.subheader("Topic Distribution")
    topic_dist = pd.Series(lda.transform(tfidf).argmax(axis=1)).value_counts().reset_index()
    topic_dist.columns = ['Topic', 'Count']
    topic_dist['Topic'] = topic_dist['Topic'].apply(lambda x: topic_names[x])  # Map topic numbers to names
    fig2 = px.pie(topic_dist, values='Count', names='Topic', title='Topic Distribution')
    st.plotly_chart(fig2)

# Community contributions
st.subheader("Community Contributions")
community_counts = filtered_df['subreddit'].value_counts().reset_index()
community_counts.columns = ['Community', 'Count']
fig3 = px.bar(community_counts, x='Community', y='Count', title='Community Contributions')
st.plotly_chart(fig3)

# User activity analysis
st.subheader("User Activity Analysis")
user_activity = filtered_df['author'].value_counts().reset_index()
user_activity.columns = ['User', 'Post Count']
fig4 = px.bar(user_activity.head(10), x='User', y='Post Count', title='Top 10 Most Active Users')
st.plotly_chart(fig4)

# Sentiment analysis
st.subheader("Sentiment Analysis")
sentiment_counts = filtered_df['sentiment_category'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig5 = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Distribution')
st.plotly_chart(fig5)

# Scatter plot: Sentiment vs. Post Score
st.subheader("Sentiment vs. Post Score")
fig6 = px.scatter(filtered_df, x='sentiment', y='score', title='Sentiment vs. Post Score', trendline='ols')
st.plotly_chart(fig6)

# Scatter plot: Post Length vs. Score
st.subheader("Post Length vs. Score")
fig7 = px.scatter(filtered_df, x='text_length', y='score', title='Post Length vs. Score', trendline='ols')
st.plotly_chart(fig7)

# Chatbot
st.header("Chatbot: Ask Questions About the Data")
user_input = st.text_input("Ask a question about the data:")
if user_input and chatbot:
    with st.spinner('Generating response...'):
        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        st.write(f"**Chatbot:** {response[0]['generated_text']}")

# Display raw data
st.header("Raw Data")
st.write(filtered_df)
