import pandas as pd

def load_jsonl(filepath):
    """Loads JSONL and extracts relevant fields."""
    df = pd.read_json(filepath, lines=True)

    # Extract fields from 'data' column
    if 'data' in df.columns:
        df = pd.json_normalize(df['data'])

    # Convert timestamp if present
    if 'created_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')

    # Drop unnecessary columns
    df = df[['timestamp', 'subreddit', 'title', 'selftext', 'score', 'num_comments']]

    return df

# Example usage
if __name__ == "__main__":
    df = load_jsonl("data/dataset.jsonl")
    
    # Basic EDA
    print("\n📊 Dataset Overview:")
    print(df.info())

    print("\n🔍 Missing Values:")
    print(df.isnull().sum())

    print("\n📈 Top 5 Subreddits:")
    print(df['subreddit'].value_counts().head())

    print("\n🔥 Top 5 Posts by Score:")
    print(df[['title', 'score']].sort_values(by='score', ascending=False).head())
