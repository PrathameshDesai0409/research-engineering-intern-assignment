# research-engineering-intern-assignment

## Overview

This project is an interactive dashboard designed to analyze and visualize social media data. It focuses on tracking how information spreads, particularly from unreliable sources. The dashboard integrates topic modeling, sentiment analysis, and a chatbot for querying insights.

## Project Structure

```
├── chatbot
│   ├── chatbot.py                # Handles chatbot interactions
│   ├── query_engine.py           # Query engine for retrieving relevant data
│   ├── vector_store.py           # Vector storage for chatbot responses
│
├── data
│   ├── dataset.jsonl              # Raw dataset containing social media posts
│   ├── preprocessed_texts.json    # Processed text data for analysis
│   ├── sentiment_results.csv      # Sentiment analysis results
│
├── faiss_index                    # Folder containing FAISS index for similarity search
│
├── models
│   ├── lda_dictionary             # Dictionary file for LDA model
│   ├── lda_model                  # Trained LDA topic model
│   ├── lda_model.expElogbeta.npy  # LDA model parameters
│   ├── lda_model.id2word          # ID-to-word mapping for LDA
│   ├── lda_model.state            # State of the LDA model
│   ├── lda_visualization.html     # HTML visualization of LDA topics
│
├── scripts
│   ├── eda.py                     # Exploratory Data Analysis (EDA)
│   ├── load_data.py                # Loads data for processing
│   ├── preprocess_text.py          # Text preprocessing steps
│   ├── sentiment_analysis.py       # Performs sentiment analysis
│   ├── topic_modeling.py           # Topic modeling using LDA
│   ├── visualize_topics.py         # Visualization of topic modeling results
│
├── app.py                          # Main application script
├── instructions.md                 # Additional setup instructions
├── README.md                       # Project documentation
```

## Features

- **Exploratory Data Analysis (EDA):** Helps understand dataset distribution.
- **Text Preprocessing:** Cleans and prepares text data.
- **Sentiment Analysis:** Analyzes sentiments (positive, negative, neutral) in social media posts.
- **Topic Modeling:** Uses Latent Dirichlet Allocation (LDA) to identify topics in the dataset.
- **Chatbot:** Allows users to query insights using FAISS-based similarity search.
- **Visualization:** Generates interactive topic modeling graphs.

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd research-engineering-intern-assignment
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run preprocessing:
   ```sh
   python scripts/preprocess_text.py
   ```
4. Perform sentiment analysis:
   ```sh
   python scripts/sentiment_analysis.py
   ```
5. Train topic model:
   ```sh
   python scripts/topic_modeling.py
   ```
6. Run the chatbot:
   ```sh
   python chatbot/chatbot.py
   ```
7. Start the main application:
   ```sh
   python app.py
   ```

## System Design & Thought Process

The project is structured around modular components for better scalability and maintainability. The key design choices include:

- **Modular Scripts:** Separate scripts for EDA, preprocessing, analysis, and visualization.
- **Efficient Storage:** FAISS index and vector storage allow fast retrieval of relevant insights.
- **LDA for Topic Modeling:** Gensim's LDA was chosen for topic extraction as it effectively identifies key themes.
- **Flask for Web Application:** The dashboard is built using Flask to allow easy deployment and API integration.

## Hosted Web Platform

The dashboard is hosted at: [Provide Deployment URL]

## Screenshots

### Dashboard View
![Dashboard Screenshot](https://github.com/user-attachments/assets/5ca773f8-9f18-4226-aafb-445a1b6adb3b)

### Sentiment Analysis Results
![Sentiment Screenshot](https://github.com/user-attachments/assets/cd5fa78a-3ec7-49ed-8c2e-dd588b154cd2)

### Topic Modeling Visualization
![Topic Screenshot]([path/to/topic_screenshot.png](https://github.com/user-attachments/assets/506f943e-7d97-440c-b3d2-3c30d5d4e449))

### Chatbot Integrations
![Chatbot](https://github.com/user-attachments/assets/0da152fa-e1b4-4505-9a16-0d6042e739c8)
## Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn, NLTK, FAISS, Gensim, Flask)
- **FAISS** for efficient similarity search
- **Gensim** for topic modeling
- **Flask** for the web dashboard
- **Matplotlib/Seaborn** for data visualization

## Future Enhancements

- Improve chatbot responses using fine-tuned language models.
- Implement real-time data fetching and analysis.
- Expand visualization capabilities for better insights.

## Contributors

- Prathamesh Desai

---

