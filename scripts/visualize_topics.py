import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import json

# Load preprocessed texts
with open("data/preprocessed_texts.json", "r", encoding="utf-8") as f:
    preprocessed_texts = json.load(f)

# Load dictionary and model (fixing filename)
dictionary = Dictionary.load("models/lda_dictionary")
lda_model = LdaModel.load("models/lda_model")

# Create corpus
corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

# Generate visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Save HTML file
pyLDAvis.save_html(vis, "models/lda_visualization.html")
print("Visualization saved to models/lda_visualization.html")
