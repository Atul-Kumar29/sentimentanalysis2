import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --- Model Initialization ---

# For Summarization (Applying the fix for the ValueError)
# We explicitly load the tokenizer and disable the "fast" version to prevent the error.
summarizer_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", use_fast=False)
summarizer = pipeline("summarization", model="ai4bharat/IndicBART", tokenizer=summarizer_tokenizer)

# For Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# For Text-to-Number Conversion
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


st.title("🇮🇳 Indic Feedback Analysis System")
st.markdown("Analyzes feedback in various Indian languages, with the tokenizer error fixed.")

# --- Data Input ---
uploaded_file = st.file_uploader("Upload Feedback CSV (with a 'feedback' column)", type=["csv"])
user_input = st.text_area("Or enter feedback comments here (one per line):")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif user_input.strip() != "":
    feedback_list = [line.strip() for line in user_input.split("\n") if line.strip()]
    df = pd.DataFrame(feedback_list, columns=["feedback"])
else:
    df = None

# --- Analysis Section ---
if df is not None:
    if 'feedback' not in df.columns:
        st.error("CSV must have a 'feedback' column or enter text manually!")
    else:
        st.success(f"Loaded {len(df)} feedback entries")

        # --- Sentiment Analysis ---
        def get_sentiment(text):
            result = sentiment_pipeline(str(text))[0]
            sentiment_map = {'Label_0': 'Negative', 'Label_1': 'Neutral', 'Label_2': 'Positive'}
            return sentiment_map.get(result['label'], 'Unknown')

        df['sentiment_label'] = df['feedback'].apply(get_sentiment)


        # --- Text Vectorization ---
        feedback_docs = df['feedback'].astype(str).tolist()
        X = embedding_model.encode(feedback_docs)


        # --- Clustering with KMeans ---
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(X)


        # --- Outlier Detection ---
        iso = IsolationForest(random_state=42, contamination=0.05)
        outliers = iso.fit_predict(X)
        df['outlier'] = outliers
        urgent_feedback = df[df['outlier'] == -1]['feedback']


        # --- Summarization ---
        try:
            summary_text = " ".join(df['feedback'].astype(str).tolist())[:2048]
            summary = summarizer(summary_text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
        except Exception:
            summary = "Summarisation model too slow or text too long in demo. Try smaller dataset."


        # --- Display Results ---
        st.subheader("1) Sentiment Distribution")
        st.bar_chart(df['sentiment_label'].value_counts())

        st.subheader("2) Word Cloud of Feedback")
        # NOTE: A font that supports Indic characters is required for this to render correctly.
        try:
            font_path = 'Nirmala.ttf' # Common font on Windows
            text = " ".join(df['feedback'].astype(str))
            wc = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except IOError:
             st.warning("Word Cloud for Indic text failed. A specific font file (like 'Nirmala.ttf') is required but was not found.")


        st.subheader("3) Feedback Clusters")
        st.write(df.groupby("cluster")['feedback'].apply(lambda x: list(x)[:5]))

        st.subheader("4) Urgent / Outlier Feedback")
        if not urgent_feedback.empty:
            st.warning("Critical feedback detected")
            st.write(urgent_feedback.head(10))
        else:
            st.info("No urgent feedback detected")

        st.subheader("5) AI-Generated Summary")
        st.write(summary)
