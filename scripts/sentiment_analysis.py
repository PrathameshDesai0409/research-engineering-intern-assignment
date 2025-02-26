import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_data(filepath):
    """Loads preprocessed JSONL dataset and extracts relevant fields."""
    try:
        df = pd.read_json(filepath, lines=True)
        
        # Extract 'data' column if present
        if 'data' in df.columns:
            df = pd.json_normalize(df['data'])

        # Convert timestamp if available
        if 'created_utc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
        else:
            df['timestamp'] = pd.NaT
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'subreddit', 'title', 'selftext', 'score', 'num_comments']
        df = df[[col for col in required_columns if col in df.columns]]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def analyze_sentiment(df):
    """Performs sentiment analysis on the title + selftext fields."""
    if df.empty:
        print("No data available for sentiment analysis.")
        return df

    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0  # Neutral if no text
        return analyzer.polarity_scores(text)['compound']

    # Combine title and selftext, handling missing values
    df['text'] = df[['title', 'selftext']].fillna('').agg(' '.join, axis=1)
    df['sentiment_score'] = df['text'].apply(get_sentiment)
    
    # Categorize sentiment
    df['sentiment'] = df['sentiment_score'].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))
    
    return df

if __name__ == "__main__":
    dataset_path = "data/dataset.jsonl"
    output_path = "data/sentiment_results.csv"
    
    print("📥 Loading dataset...")
    df = load_data(dataset_path)
    
    if not df.empty:
        print("🧠 Performing sentiment analysis...")
        df = analyze_sentiment(df)

        print("💾 Saving results...")
        df.to_csv(output_path, index=False)
        
        print("\n📊 Sentiment Distribution:")
        print(df['sentiment'].value_counts())

        print("\n🔥 Top Positive Posts:")
        print(df[df['sentiment'] == 'positive'][['title', 'sentiment_score']].head())

        print("\n😡 Top Negative Posts:")
        print(df[df['sentiment'] == 'negative'][['title', 'sentiment_score']].head())
    else:
        print("❌ No data found. Exiting...")
