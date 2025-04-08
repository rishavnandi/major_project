import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import plotly.express as px
import numpy as np
import re  # For basic cleaning

# --- NLTK Setup ---
# Download required NLTK data (run only once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Constants and Configuration ---
DATA_FILE = 'short_dataset.csv'  # Make sure this file is in the same directory

ASPECT_KEYWORDS = {
    'Behaviour/Staff': ['staff', 'nurse', 'doctor', 'behavior', 'behaviour', 'rude', 'polite', 'friendly', 'cooperative', 'attitude', 'empathy', 'unprofessional', 'professional', 'caring', 'kind', 'attentive', 'irresponsible', 'helpful', 'supportive', 'compassion', 'communication'],
    'Cost/Billing': ['cost', 'price', 'expensive', 'cheap', 'affordable', 'bill', 'billing', 'charges', 'money', 'fees', 'rate', 'payment', 'insurance', 'cashless', 'amount', 'paid', 'pay', 'value', 'financial', 'loot', 'overpriced', 'reasonable'],
    'Cleanliness/Hygiene': ['clean', 'dirty', 'hygiene', 'hygienic', 'neat', 'tidy', 'messy', 'sanitized', 'sanitation', 'unhygienic', 'spotless', 'cleanliness'],
    'Wait Times/Efficiency': ['wait', 'waiting', 'queue', 'delay', 'late', 'slow', 'time', 'hours', 'quick', 'fast', 'efficient', 'prompt', 'appointment', 'schedule', 'long', 'process', 'admit', 'admission', 'discharge'],
    'Facilities/Amenities': ['facilities', 'facility', 'equipment', 'beds', 'room', 'ward', 'infrastructure', 'amenities', 'ac', 'toilet', 'washroom', 'bathroom', 'parking', 'lift', 'ambiance', 'environment', 'comfortable', 'canteen', 'food', 'space', 'water', 'building'],
    'Treatment/Diagnosis': ['treatment', 'diagnosis', 'care', 'medical', 'surgery', 'operation', 'procedure', 'medicine', 'medication', 'health', 'heal', 'cure', 'recover', 'quality', 'effective', 'skill', 'expertise', 'accurate', 'improve', 'condition', 'pain', 'relief']
}

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# --- Helper Functions ---


def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation (optional, VADER handles some)
    # text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_data  # Cache the data loading and processing
def load_and_process_data(file_path):
    """Loads data, performs sentiment and aspect analysis."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(
            f"Error: Data file '{file_path}' not found. Please place it in the same directory.")
        return pd.DataFrame(), pd.DataFrame()  # Return empty dataframes

    # Drop rows with missing text or title
    df.dropna(subset=['text', 'title'], inplace=True)
    df['text'] = df['text'].astype(str)  # Ensure text is string
    df['title'] = df['title'].astype(str)

    # --- Overall Sentiment ---
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['overall_sentiment_score'] = df['cleaned_text'].apply(
        lambda text: analyzer.polarity_scores(text)['compound'])

    # --- Aspect-Based Sentiment ---
    aspect_columns = {}
    for aspect in ASPECT_KEYWORDS:
        aspect_columns[f'{aspect}_mentioned'] = []
        aspect_columns[f'{aspect}_sentiment'] = []

    for review_text in df['text']:  # Use original text for sentence tokenization
        if not isinstance(review_text, str):  # Handle potential non-string data
            for aspect in ASPECT_KEYWORDS:
                aspect_columns[f'{aspect}_mentioned'].append(False)
                aspect_columns[f'{aspect}_sentiment'].append(np.nan)
            continue

        review_lower = review_text.lower()
        sentences = sent_tokenize(review_text)
        review_aspect_sentiments = {aspect: [] for aspect in ASPECT_KEYWORDS}
        aspects_found_in_review = {aspect: False for aspect in ASPECT_KEYWORDS}

        for aspect, keywords in ASPECT_KEYWORDS.items():
            sentences_with_aspect = []
            for keyword in keywords:
                if keyword in review_lower:
                    aspects_found_in_review[aspect] = True
                    # Find sentences containing the keyword
                    for sentence in sentences:
                        if keyword in sentence.lower() and sentence not in sentences_with_aspect:
                            sentences_with_aspect.append(sentence)

            # Calculate sentiment for relevant sentences
            if sentences_with_aspect:
                aspect_sentence_text = " ".join(sentences_with_aspect)
                sentiment = analyzer.polarity_scores(
                    aspect_sentence_text)['compound']
                review_aspect_sentiments[aspect].append(sentiment)

        # Append results for this review
        for aspect in ASPECT_KEYWORDS:
            aspect_columns[f'{aspect}_mentioned'].append(
                aspects_found_in_review[aspect])
            sentiments = review_aspect_sentiments[aspect]
            if sentiments:
                aspect_columns[f'{aspect}_sentiment'].append(
                    np.mean(sentiments))
            else:
                aspect_columns[f'{aspect}_sentiment'].append(
                    np.nan)  # Use NaN if aspect not mentioned

    # Add aspect columns to DataFrame
    for col_name, data in aspect_columns.items():
        df[col_name] = data

    # --- Aggregate Data by Hospital ---
    hospital_agg = df.groupby('title').agg(
        avg_google_score=('totalScore', 'mean'),
        avg_overall_sentiment=('overall_sentiment_score', 'mean'),
        review_count=('text', 'size'),
        # Add aggregation for each aspect
        **{f'{aspect}_mention_pct': (f'{aspect}_mentioned', lambda x: x.mean() * 100) for aspect in ASPECT_KEYWORDS},
        # Calculates mean ignoring NaNs
        **{f'{aspect}_avg_sentiment': (f'{aspect}_sentiment', 'mean') for aspect in ASPECT_KEYWORDS}
    ).reset_index()

    # Rename columns for clarity in the dashboard
    hospital_agg.rename(columns={'title': 'Hospital Name',
                                 'avg_google_score': 'Average Google Score',
                                 'avg_overall_sentiment': 'Average Review Sentiment (VADER)',
                                 'review_count': 'Number of Reviews'}, inplace=True)

    # Rename aspect columns for better display
    for aspect in ASPECT_KEYWORDS:
        hospital_agg.rename(columns={
            f'{aspect}_mention_pct': f'{aspect} Mention (%)',
            f'{aspect}_avg_sentiment': f'{aspect} Average Sentiment'
        }, inplace=True)

    return df, hospital_agg

# --- Streamlit App Layout ---


st.set_page_config(
    layout="wide", page_title="Hospital Review Sentiment Analysis")

st.title("🏥 Hospital Review Aspect-Based Sentiment Analysis")
st.markdown("""
Welcome to the Hospital Review Analysis Dashboard.
This dashboard analyzes customer reviews to understand overall sentiment and sentiment towards specific aspects like *Cleanliness*, *Cost*, *Staff Behaviour*, etc.
Sentiment scores range from -1 (very negative) to +1 (very positive).
""")

# --- Load Data ---
raw_df, hospital_data = load_and_process_data(DATA_FILE)

if raw_df.empty or hospital_data.empty:
    st.stop()  # Stop execution if data loading failed

# --- Sidebar ---
st.sidebar.header("Navigation")
analysis_mode = st.sidebar.radio(
    "Select Analysis View", ("Overall Analysis", "Hospital Specific Analysis", "Aspect Ranking"))

st.sidebar.header("Data Overview")
st.sidebar.metric("Total Hospitals Analyzed", hospital_data.shape[0])
st.sidebar.metric("Total Reviews Analyzed", raw_df.shape[0])
st.sidebar.markdown("---")
st.sidebar.markdown("Aspects Analyzed:")
for aspect in ASPECT_KEYWORDS.keys():
    st.sidebar.markdown(f"- {aspect}")

# --- Main Content ---

# === Overall Analysis ===
if analysis_mode == "Overall Analysis":
    st.header("Overall Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Google Scores")
        fig_google = px.histogram(hospital_data, x='Average Google Score',
                                  nbins=10, title='Hospital Average Google Score Distribution')
        fig_google.update_layout(bargap=0.1)
        st.plotly_chart(fig_google, use_container_width=True)

    with col2:
        st.subheader("Distribution of Overall Review Sentiment (VADER)")
        fig_vader = px.histogram(hospital_data, x='Average Review Sentiment (VADER)',
                                 nbins=20, title='Hospital Average VADER Score Distribution')
        fig_vader.update_layout(bargap=0.1)
        st.plotly_chart(fig_vader, use_container_width=True)

    st.subheader("Correlation: Google Score vs. Calculated Sentiment")
    # Calculate correlation
    correlation = hospital_data['Average Google Score'].corr(
        hospital_data['Average Review Sentiment (VADER)'])

    st.metric("Pearson Correlation Coefficient", f"{correlation:.3f}")
    st.markdown("""
    This correlation measures the linear relationship between the average Google star rating given by users and the average sentiment score calculated from the review text using VADER.
    A value closer to 1 indicates a strong positive relationship (higher Google scores tend to correspond with more positive text sentiment).
    A value closer to -1 indicates a strong negative relationship.
    A value near 0 indicates little to no linear relationship.
    *Note: This is not an 'accuracy' score in the ML sense, but an indicator of alignment between the provided score and the calculated text sentiment.*
    """)

    fig_corr = px.scatter(hospital_data, x='Average Google Score', y='Average Review Sentiment (VADER)',
                          hover_data=['Hospital Name', 'Number of Reviews'],
                          title='Average Google Score vs. Average VADER Sentiment per Hospital',
                          trendline='ols',  # Add a regression line
                          labels={'Average Google Score': 'Avg. Google Score (Stars)', 'Average Review Sentiment (VADER)': 'Avg. Review Sentiment (VADER Score)'})
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Overall Hospital Data")
    st.dataframe(hospital_data.round(3), use_container_width=True)


# === Hospital Specific Analysis ===
elif analysis_mode == "Hospital Specific Analysis":
    st.header("Hospital Specific Analysis")
    hospital_list = sorted(hospital_data['Hospital Name'].unique())
    selected_hospital = st.selectbox("Select a Hospital", hospital_list)

    if selected_hospital:
        hospital_info = hospital_data[hospital_data['Hospital Name']
                                      == selected_hospital].iloc[0]
        raw_hospital_reviews = raw_df[raw_df['title'] == selected_hospital]

        st.subheader(f"Analysis for: {selected_hospital}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Google Score",
                    f"{hospital_info['Average Google Score']:.2f} ⭐")
        col2.metric("Average Review Sentiment (VADER)",
                    f"{hospital_info['Average Review Sentiment (VADER)']:.3f}")
        col3.metric("Number of Reviews",
                    f"{hospital_info['Number of Reviews']}")

        st.markdown("---")
        st.subheader("Aspect Sentiment Breakdown")
        st.markdown(
            "Showing the percentage of reviews mentioning an aspect and the average sentiment score for those mentions.")

        aspect_data = []
        for aspect in ASPECT_KEYWORDS.keys():
            mention_pct = hospital_info.get(
                f'{aspect} Mention (%)', 0)  # Use .get for safety
            avg_sentiment = hospital_info.get(
                f'{aspect} Average Sentiment', np.nan)
            if pd.notna(avg_sentiment):  # Only include aspects that were mentioned
                aspect_data.append(
                    {'Aspect': aspect, 'Mention (%)': mention_pct, 'Average Sentiment': avg_sentiment})
            else:
                # Show 0 if never mentioned with sentiment
                aspect_data.append(
                    {'Aspect': aspect, 'Mention (%)': mention_pct, 'Average Sentiment': 0.0})

        if aspect_data:
            aspect_df = pd.DataFrame(aspect_data)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_mention = px.bar(aspect_df, x='Aspect', y='Mention (%)', title='Aspect Mention Frequency',
                                     color='Aspect', labels={'Mention (%)': 'Percentage of Reviews Mentioning Aspect'})
                fig_mention.update_layout(
                    xaxis_title=None, yaxis_title="Mention Percentage (%)", showlegend=False)
                st.plotly_chart(fig_mention, use_container_width=True)

            with col_chart2:
                # Ensure sentiment scale is appropriate (-1 to 1)
                min_sentiment = aspect_df['Average Sentiment'].min()
                max_sentiment = aspect_df['Average Sentiment'].max()
                # Ensure range covers potential values
                range_sentiment = [
                    min(min_sentiment, -0.1) - 0.1, max(max_sentiment, 0.1) + 0.1]

                fig_sentiment = px.bar(aspect_df, x='Aspect', y='Average Sentiment', title='Average Sentiment per Aspect (when mentioned)',
                                       color='Average Sentiment',
                                       color_continuous_scale=px.colors.diverging.RdYlGn,  # Red-Yellow-Green scale
                                       # Fix scale from -1 to 1
                                       range_color=[-1, 1],
                                       labels={'Average Sentiment': 'Average VADER Score'})
                fig_sentiment.update_layout(xaxis_title=None, yaxis_title="Average Sentiment Score", yaxis=dict(
                    range=[-1.1, 1.1]))  # Fixed y-axis range
                fig_sentiment.update_coloraxes(
                    showscale=False)  # Hide color bar if desired
                st.plotly_chart(fig_sentiment, use_container_width=True)

            st.dataframe(aspect_df.round(3), use_container_width=True)

        else:
            st.warning("No aspects mentioned or analyzed for this hospital.")

        # Display Sample Reviews
        st.subheader("Sample Reviews")
        show_reviews = st.checkbox(
            "Show sample reviews for this hospital?", value=False)
        if show_reviews:
            st.dataframe(raw_hospital_reviews[['text', 'totalScore', 'overall_sentiment_score']].head(
                10), use_container_width=True)


# === Aspect Ranking ===
elif analysis_mode == "Aspect Ranking":
    st.header("Hospital Ranking by Aspect Sentiment")
    aspect_list = list(ASPECT_KEYWORDS.keys())
    selected_aspect = st.selectbox(
        "Select an Aspect to Rank Hospitals By", aspect_list)

    if selected_aspect:
        sentiment_col = f'{selected_aspect} Average Sentiment'
        mention_col = f'{selected_aspect} Mention (%)'

        # Filter hospitals where the aspect was mentioned and sentiment calculated
        ranked_hospitals = hospital_data[hospital_data[sentiment_col].notna()].copy(
        )

        # Sort by the selected aspect's average sentiment
        ranked_hospitals.sort_values(
            by=sentiment_col, ascending=False, inplace=True)

        st.subheader(
            f"Hospitals Ranked by Average Sentiment for '{selected_aspect}'")
        st.markdown(
            f"Showing hospitals where '{selected_aspect}' was mentioned, sorted by the average sentiment expressed towards it.")

        # Select and rename columns for display
        display_cols = ['Hospital Name', sentiment_col,
                        mention_col, 'Number of Reviews', 'Average Google Score']
        display_df = ranked_hospitals[display_cols].round(3)

        st.dataframe(display_df, use_container_width=True)


st.markdown("---")
st.markdown("Dashboard developed by [CSEMP144]")
st.markdown("Data Source: [Google Reviews]")
st.markdown("Data Collection Methods: [Web Scraping, Free APIs, Manual Entry]")
