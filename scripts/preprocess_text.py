import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

raw_texts = [
    "Social media spreads information fast. Some of it is unreliable!",
    "AI and machine learning are changing the way we analyze data.",
    "Fake news spreads misinformation and influences opinions."
]

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Preprocess all texts
preprocessed_texts = [preprocess_text(text) for text in raw_texts]

# Save to JSON
output_file = "data/preprocessed_texts.json"
with open(output_file, "w") as f:
    json.dump(preprocessed_texts, f)

print(f"Preprocessed data saved to {output_file}")
