import pandas as pd

def load_jsonl(filepath):
    """Loads a JSONL file into a Pandas DataFrame."""
    df = pd.read_json(filepath, lines=True)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
    
    # Drop rows where 'text' is missing and fill NaN values
    if 'text' in df.columns:
        df = df.dropna(subset=['text']).fillna("")
    
    # Normalize lists (hashtags, urls, mentions)
    for col in ['hashtags', 'urls', 'mentions']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return df

# Example usage
if __name__ == "__main__":
    df = load_jsonl("data/dataset.jsonl")
    print(df.head())
