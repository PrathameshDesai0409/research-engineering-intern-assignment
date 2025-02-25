import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_data(filepath):
    """Loads preprocessed JSONL dataset."""
    df = pd.read_json(filepath, lines=True)
    
    # Extract 'data' column if present
    if 'data' in df.columns:
        df = pd.json_normalize(df['data'])

    # Convert timestamp
    if 'created_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')

    # Select relevant columns
    df = df[['timestamp', 'subreddit', 'title', 'selftext', 'score', 'num_comments']]

    return df

def analyze_sentiment(df):
    """Performs sentiment analysis on the title + selftext fields."""
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0  # Neutral if no text
        
        sentiment_score = analyzer.polarity_scores(text)['compound']
        return sentiment_score

    # Apply sentiment analysis
    df['text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
    df['sentiment_score'] = df['text'].apply(get_sentiment)

    # Categorize sentiment
    df['sentiment'] = df['sentiment_score'].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))

    return df

if __name__ == "__main__":
    df = load_data("data/dataset.jsonl")
    
    df = analyze_sentiment(df)

    # Save processed data
    df.to_csv("data/sentiment_results.csv", index=False)

    # Print summary
    print("\n📊 Sentiment Distribution:")
    print(df['sentiment'].value_counts())

    print("\n🔥 Top Positive Posts:")
    print(df[df['sentiment'] == 'positive'][['title', 'sentiment_score']].head())

    print("\n😡 Top Negative Posts:")
    print(df[df['sentiment'] == 'negative'][['title', 'sentiment_score']].head())
