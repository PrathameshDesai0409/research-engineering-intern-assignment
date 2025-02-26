import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import faiss
import numpy as np
import json
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from chatbot.chatbot import query_engine  # Import chatbot function

st.set_page_config(page_title="Social Media Analytics Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Data Analytics", "💬 Chatbot"])

if page == "📊 Data Analytics":
    st.title("📊 Social Media Analytics Dashboard")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data:")
        st.write(df.head())

        # Sentiment Analysis Visualization
        if "sentiment" in df.columns:
            st.write("### Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.countplot(x=df["sentiment"], palette="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No sentiment data found in the dataset!")

        # Topic Modeling (LDA)
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

    st.write("👈 Upload a dataset to start exploring!")

elif page == "💬 Chatbot":
    st.title("💬 AI Chatbot")

    FAISS_INDEX_PATH = "faiss_index/index.faiss"

    # Load FAISS Index
    @st.cache_resource
    def load_faiss_index():
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error("FAISS index not found! Run `vector_store.py` first.")
            return None
        return faiss.read_index(FAISS_INDEX_PATH)

    # Chatbot Interface
    st.write("Ask me anything related to the dataset!")

    user_query = st.text_input("Enter your question:")

    # Load FAISS index
    index = load_faiss_index()

    if user_query:
        if index is not None:
            st.write("🔍 Searching for relevant documents...")
            response = query_engine(user_query, top_k=3)  # Get top 3 results

            st.subheader("📌 Answer:")
            for res in response:
                st.write(f"- {res}")  # Display chatbot responses as a list
        else:
            st.error("⚠️ Error loading FAISS index. Try running `vector_store.py` again.")
