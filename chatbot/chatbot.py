import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from chatbot.query_engine import query_engine  # Import chatbot query engine

# Set Streamlit page config
st.set_page_config(page_title="Social Media Analytics Dashboard", layout="wide")

st.title("📊 Social Media Analytics Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.write(df.head())

    # Sentiment Distribution
    if "sentiment" in df.columns:
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.countplot(x=df["sentiment"], palette="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No sentiment data found in the dataset!")

    # Topic Modeling Visualization
    try:
        dictionary = Dictionary.load("models/lda_dictionary")
        lda_model = LdaModel.load("models/lda_model")

        corpus = [dictionary.doc2bow(text.split()) for text in df["text"].astype(str)]
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        lda_html_path = "models/lda_visualization.html"
        pyLDAvis.save_html(vis, lda_html_path)

        st.write("### Topic Modeling Visualization")
        st.markdown(f"[Click here to view LDA visualization]({lda_html_path})")
    except FileNotFoundError:
        st.warning("LDA Model not found. Please train the model first.")
    except KeyError:
        st.warning("No text column found for topic modeling!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---- Chatbot Section ----
st.write("## 🤖 Chat with the AI")
user_query = st.text_input("Ask the chatbot about social media trends, AI, or misinformation:")

if user_query:
    response = query_engine(user_query)
    st.write("**Chatbot Response:**")
    st.write(response[0])  # Show the first response

st.write("👈 Upload a dataset to start exploring!")
