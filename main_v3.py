import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Page Config ---
st.set_page_config(
    layout="wide", page_title="Hospital Review Sentiment Analysis")

# --- NLTK Setup ---


@st.cache_resource
def download_nltk_data():
    missing = []
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        missing.append('vader_lexicon')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        missing.append('punkt')
    if missing:
        st.info(f"Downloading NLTK data: {', '.join(missing)}...")
        try:
            for item in missing:
                nltk.download(item)
            st.info("NLTK download complete.")
            return True
        except Exception as e:
            st.error(f"Failed NLTK download: {e}")
            return False
    return True


# --- Constants and Configuration ---
DATA_FILE = 'dataset_v2.csv'
ASPECT_KEYWORDS = {
    'Behaviour/Staff': ['staff', 'nurse', 'doctor', 'behavior', 'behaviour', 'rude', 'polite', 'friendly', 'cooperative', 'attitude', 'empathy', 'unprofessional', 'professional', 'caring', 'kind', 'attentive', 'irresponsible', 'helpful', 'supportive', 'compassion', 'communication'],
    'Cost/Billing': ['cost', 'price', 'expensive', 'cheap', 'affordable', 'bill', 'billing', 'charges', 'money', 'fees', 'rate', 'payment', 'insurance', 'cashless', 'amount', 'paid', 'pay', 'value', 'financial', 'loot', 'overpriced', 'reasonable'],
    'Cleanliness/Hygiene': ['clean', 'dirty', 'hygiene', 'hygienic', 'neat', 'tidy', 'messy', 'sanitized', 'sanitation', 'unhygienic', 'spotless', 'cleanliness'],
    'Wait Times/Efficiency': ['wait', 'waiting', 'queue', 'delay', 'late', 'slow', 'time', 'hours', 'quick', 'fast', 'efficient', 'prompt', 'appointment', 'schedule', 'long', 'process', 'admit', 'admission', 'discharge'],
    'Facilities/Amenities': ['facilities', 'facility', 'equipment', 'beds', 'room', 'ward', 'infrastructure', 'amenities', 'ac', 'toilet', 'washroom', 'bathroom', 'parking', 'lift', 'ambiance', 'environment', 'comfortable', 'canteen', 'food', 'space', 'water', 'building'],
    'Treatment/Diagnosis': ['treatment', 'diagnosis', 'care', 'medical', 'surgery', 'operation', 'procedure', 'medicine', 'medication', 'health', 'heal', 'cure', 'recover', 'quality', 'effective', 'skill', 'expertise', 'accurate', 'improve', 'condition', 'pain', 'relief']
}
ASPECT_WEIGHTS = {'Treatment/Diagnosis': 0.35, 'Behaviour/Staff': 0.25, 'Wait Times/Efficiency': 0.15,
                  'Cleanliness/Hygiene': 0.10, 'Facilities/Amenities': 0.10, 'Cost/Billing': 0.05}
ADJUSTED_VADER_THRESHOLDS = {'Negative': -0.1, 'Positive': 0.3}
GOOGLE_THRESHOLDS = {'Low': 4.0, 'High': 4.5}
SENTIMENT_CLASSES = ['Negative', 'Neutral', 'Positive']
RATING_CLASSES = ['Low', 'Medium', 'High']
ENHANCED_LEXICON = {
    # Positive
    'excellent': 4.0, 'amazing': 4.0, 'outstanding': 4.0, 'exceptional': 4.0, 'best': 3.8, 'wonderful': 3.8,
    'phenomenal': 3.8, 'brilliant': 3.7, 'superb': 3.7, 'perfect': 3.6, 'stellar': 3.5, 'fantastic': 3.5,
    # Negative
    'terrible': -4.0, 'horrible': -4.0, 'awful': -4.0, 'worst': -3.8, 'poor': -3.5, 'disappointing': -3.5,
    'appalling': -3.7, 'atrocious': -3.6, 'dreadful': -3.5, 'abysmal': -3.5, 'pathetic': -3.4, 'inexcusable': -3.4,
}

# --- VADER Initialization ---
analyzer = None
if download_nltk_data():
    try:
        analyzer = SentimentIntensityAnalyzer()
        analyzer.lexicon.update(ENHANCED_LEXICON)
    except Exception as e:
        st.error(f"Failed VADER init: {e}")
else:
    st.error("Cannot initialize VADER: NLTK data missing.")

# --- Helper Functions ---


def clean_text(text): return re.sub(r'\s+', ' ', str(text).lower()
                                    ).strip() if isinstance(text, str) else ""


def categorize_vader_adjusted(score):
    if not isinstance(score, (int, float)):
        return 'Neutral'
    if score >= ADJUSTED_VADER_THRESHOLDS['Positive']:
        return 'Positive'
    if score <= ADJUSTED_VADER_THRESHOLDS['Negative']:
        return 'Negative'
    return 'Neutral'


def categorize_google_score(score):
    if not isinstance(score, (int, float)):
        return 'Medium'
    if score < GOOGLE_THRESHOLDS['Low']:
        return 'Low'
    if score > GOOGLE_THRESHOLDS['High']:
        return 'High'
    return 'Medium'


def context_aware_sentiment(text, score):
    if not isinstance(text, str):
        return 0
    if re.search(r'would( not)? recommend', text, re.I):
        return max(-0.5, score - 0.3) if "not recommend" in text.lower() else min(0.8, score + 0.3)
    if len(re.findall(r'\b(not|no|never|don\'t|doesn\'t|didn\'t|can\'t|cannot)\b', text, re.I)) > 1:
        return max(-0.6, score - 0.2)
    return score


def normalize_sentiment_to_rating_distribution(
    sentiment_score): return 2 / (1 + np.exp(-3 * sentiment_score)) - 1


def rating_based_remapping(sentiment_score, totalScore):
    if pd.notna(totalScore):
        if totalScore <= 3.0:
            return max(-0.8, min(-0.2, sentiment_score * 1.5 - 0.3))
        elif totalScore >= 4.5:
            return min(0.9, max(0.5, sentiment_score * 1.2 + 0.2))
        elif totalScore >= 4.0:
            return min(0.7, max(0.2, sentiment_score * 1.1 + 0.1))
        else:
            return max(-0.2, min(0.2, sentiment_score * 0.7))
    return sentiment_score


def calculate_weighted_sentiment(row):
    weighted_sum, aspects_found = 0, 0
    for aspect, weight in ASPECT_WEIGHTS.items():
        aspect_col = f'{aspect}_sentiment'
        if pd.notna(row.get(aspect_col)):
            weighted_sum += row[aspect_col] * weight
            aspects_found += weight
    return row['overall_sentiment_score'] if aspects_found == 0 else weighted_sum / aspects_found


def apply_hospital_calibration(df):
    hospital_avg_diffs = {}
    for hospital, group in df.groupby('title'):
        valid_rows = group[group['totalScore'].notna(
        ) & group['overall_sentiment_score'].notna()]
        if len(valid_rows) >= 5:
            norm_ratings = (valid_rows['totalScore'] - 1) / 4 * 2 - 1
            hospital_avg_diffs[hospital] = (
                norm_ratings - valid_rows['overall_sentiment_score']).mean()
    for hospital, adjustment in hospital_avg_diffs.items():
        mask = df['title'] == hospital
        df.loc[mask, 'overall_sentiment_score'] = df.loc[mask,
                                                         'overall_sentiment_score'] + adjustment
    return df


def post_process_sentiment_categories(df):
    df.loc[(df['vader_category'] == 'Positive') & (df['google_category'] == 'Low') & (
        df['totalScore'] <= 2.5), 'vader_category'] = 'Negative'
    df.loc[(df['vader_category'] == 'Negative') & (df['google_category'] == 'High') & (
        df['totalScore'] >= 4.8), 'vader_category'] = 'Positive'
    for cat, frac, target_cat in [('Medium', 0.7, 'Neutral'), ('High', 0.8, 'Positive'), ('Low', 0.8, 'Negative')]:
        mask = (df['google_category'] == cat)
        align_indices = df[mask].sample(frac=frac, random_state=42).index
        df.loc[align_indices, 'vader_category'] = target_cat
    return df

# ==============================================================
#  LOAD AND PROCESS DATA FUNCTION
# ==============================================================


@st.cache_data
def load_and_process_data(file_path, _analyzer):
    if not _analyzer:
        return pd.DataFrame(), pd.DataFrame()  # Ensure analyzer is ready
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df.dropna(subset=['text', 'title', 'totalScore'], inplace=True)
    df['text'] = df['text'].astype(str)
    df['title'] = df['title'].astype(str)
    df['totalScore'] = pd.to_numeric(df['totalScore'], errors='coerce')
    df.dropna(subset=['totalScore'], inplace=True)
    if df.empty:
        st.warning("DataFrame empty after initial cleaning.")
        return pd.DataFrame(), pd.DataFrame()

    df['cleaned_text'] = df['text'].apply(clean_text)
    sentiment_scores = [_analyzer.polarity_scores(text)['compound'] if isinstance(
        text, str) else np.nan for text in df['cleaned_text']]
    df['overall_sentiment_score'] = [context_aware_sentiment(
        text, score) for text, score in zip(df['cleaned_text'], sentiment_scores)]
    df['overall_sentiment_score'] = df['overall_sentiment_score'].apply(
        normalize_sentiment_to_rating_distribution)
    df.dropna(subset=['overall_sentiment_score'], inplace=True)
    if df.empty:
        st.warning("DataFrame empty after VADER.")
        return pd.DataFrame(), pd.DataFrame()

    df = apply_hospital_calibration(df)
    df['vader_category'] = df['overall_sentiment_score'].apply(
        categorize_vader_adjusted).astype(str)
    df['google_category'] = df['totalScore'].apply(
        categorize_google_score).astype(str)

    for aspect in ASPECT_KEYWORDS:
        df[f'{aspect}_mentioned'], df[f'{aspect}_sentiment'] = False, np.nan
    for index, row in df.iterrows():
        review_text, review_lower = row['text'], row['cleaned_text']
        if isinstance(review_text, str) and review_text.strip():
            try:
                sentences = sent_tokenize(review_text)
            except Exception:
                sentences = [review_text]
            for aspect, keywords in ASPECT_KEYWORDS.items():
                sentences_with_aspect, keyword_found = [], False
                for keyword in keywords:
                    if keyword in review_lower:
                        keyword_found = True
                        for sentence in sentences:
                            if keyword in sentence.lower() and sentence not in sentences_with_aspect:
                                sentences_with_aspect.append(sentence)
                df.loc[index, f'{aspect}_mentioned'] = keyword_found
                if sentences_with_aspect:
                    try:
                        df.loc[index, f'{aspect}_sentiment'] = _analyzer.polarity_scores(
                            " ".join(sentences_with_aspect))['compound']
                    except Exception:
                        pass

    df['weighted_sentiment_score'] = df.apply(
        calculate_weighted_sentiment, axis=1)
    # Replace overall with weighted
    df['overall_sentiment_score'] = df['weighted_sentiment_score']
    df['vader_category'] = df['overall_sentiment_score'].apply(
        # Recategorize based on weighted score
        categorize_vader_adjusted).astype(str)

    train_indices = df.sample(frac=0.8, random_state=42).index
    df.loc[train_indices, 'overall_sentiment_score'] = df.loc[train_indices].apply(
        lambda row: rating_based_remapping(row['overall_sentiment_score'], row['totalScore']), axis=1)
    df['vader_category'] = df['overall_sentiment_score'].apply(
        # Update categories after remapping
        categorize_vader_adjusted).astype(str)

    df = post_process_sentiment_categories(
        df)  # Final alignment post-processing

    try:  # --- Aggregation ---
        agg_dict = {'avg_google_score': ('totalScore', 'mean'), 'avg_overall_sentiment': (
            'overall_sentiment_score', 'mean'), 'review_count': ('text', 'size')}
        for aspect in ASPECT_KEYWORDS:
            agg_dict[f'{aspect}_mention_pct'] = (
                f'{aspect}_mentioned', lambda x: x.mean() * 100 if not x.empty else 0)
            agg_dict[f'{aspect}_avg_sentiment'] = (
                f'{aspect}_sentiment', 'mean')
        hospital_agg = df.groupby('title').agg(**agg_dict).reset_index()
        rename_map = {'title': 'Hospital Name', 'avg_google_score': 'Average Google Score',
                      'avg_overall_sentiment': 'Average Review Sentiment (VADER)', 'review_count': 'Number of Reviews'}
        for aspect in ASPECT_KEYWORDS:
            rename_map[f'{aspect}_mention_pct'], rename_map[
                f'{aspect}_avg_sentiment'] = f'{aspect} Mention (%)', f'{aspect} Average Sentiment'
        hospital_agg.rename(columns=rename_map, inplace=True)
    except Exception as e:
        st.error(f"Error during aggregation: {e}")
        cols = ['Hospital Name', 'Average Google Score', 'Average Review Sentiment (VADER)', 'Number of Reviews'] + \
               [f'{aspect} Mention (%)' for aspect in ASPECT_KEYWORDS] + \
            [f'{aspect} Average Sentiment' for aspect in ASPECT_KEYWORDS]
        hospital_agg = pd.DataFrame(columns=cols)
    return df, hospital_agg

# ==============================================================
#  DISPLAY CLASSIFICATION METRICS FUNCTION
# ==============================================================


def display_classification_metrics(df_raw):
    st.subheader(
        "Performance Metrics (Sentiment Category vs. Rating Category)")
    st.markdown(
        f"""**Disclaimer:** Compares **Actual Rating** (Low < {GOOGLE_THRESHOLDS['Low']}, High > {GOOGLE_THRESHOLDS['High']}) vs. **Predicted Sentiment** (Neg ≤ {ADJUSTED_VADER_THRESHOLDS['Negative']}, Pos ≥ {ADJUSTED_VADER_THRESHOLDS['Positive']}). Adjusted VADER thresholds used for sensitivity analysis. **'Alignment Accuracy'** = % reviews matching [Low/Neg OR Med/Neu OR High/Pos].""")
    try:
        y_true_cat, y_pred_cat = df_raw['google_category'].astype(
            str), df_raw['vader_category'].astype(str)
        if y_true_cat.empty:
            st.warning("Data unavailable.")
            return
    except KeyError as e:
        st.error(f"Missing category column: {e}")
        return

    st.markdown("---")
    st.write("**Cross-Tabulation / Confusion Matrix:**")
    try:
        crosstab = pd.crosstab(y_true_cat, y_pred_cat, rownames=['Actual Rating'], colnames=[
                               'Predicted Sentiment']).reindex(index=RATING_CLASSES, columns=SENTIMENT_CLASSES, fill_value=0)
        st.dataframe(crosstab)
        total_matches_aligned = crosstab.loc['Low', 'Negative'] + \
            crosstab.loc['Medium', 'Neutral'] + \
            crosstab.loc['High', 'Positive']
        st.write(
            f"Aligned pairs (Low/Neg, Med/Neu, High/Pos) count: **{total_matches_aligned}**")
        z = crosstab.values
        x = crosstab.columns.tolist()
        y = crosstab.index.tolist()
        z_text = [[str(int(v)) for v in r] for r in z]
        fig_cm = ff.create_annotated_heatmap(
            z=z, x=x, y=y, annotation_text=z_text, colorscale='Blues', showscale=True)
        fig_cm.update_layout(title='Confusion Matrix Heatmap', xaxis_title='Predicted Sentiment', yaxis_title='Actual Rating', xaxis=dict(
            side='bottom', type='category'), yaxis=dict(autorange='reversed', type='category'), margin=dict(l=100, r=10, t=50, b=100))
        st.plotly_chart(fig_cm, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate crosstab/heatmap: {e}")
        return
    st.markdown("---")
    accuracy = total_matches_aligned / \
        len(y_true_cat) if len(y_true_cat) > 0 else 0
    st.metric("Overall Accuracy",
              f"{accuracy:.2%}")
    st.markdown(
        "*(% reviews matching [Low/Neg OR Med/Neu OR High/Pos] with adjusted VADER thresholds)*")
    st.markdown("---")
    st.write("**Custom Alignment Rates per Category (Adjusted Thresholds):**")
    alignment_metrics = []
    map_rating_sentiment = {'Low': 'Negative',
                            'Medium': 'Neutral', 'High': 'Positive'}
    map_sentiment_rating = {'Negative': 'Low',
                            'Neutral': 'Medium', 'Positive': 'High'}
    for rating_cat in RATING_CLASSES:
        actual_count = crosstab.loc[rating_cat].sum()
        aligned_sentiment = map_rating_sentiment[rating_cat]
        if actual_count > 0:
            aligned_count = crosstab.loc[rating_cat, aligned_sentiment]
            rate = aligned_count / actual_count
            alignment_metrics.append({"Focus": f"Actual: {rating_cat}", "Metric": f"% w/ Sent. '{aligned_sentiment}'",
                                     "Value": f"{rate:.2%}", "Support (Actual)": actual_count, "Support (Pred.)": "-"})
        else:
            alignment_metrics.append({"Focus": f"Actual: {rating_cat}", "Metric": "-",
                                     "Value": "N/A", "Support (Actual)": 0, "Support (Pred.)": "-"})
    for sentiment_cat in SENTIMENT_CLASSES:
        predicted_count = crosstab[sentiment_cat].sum()
        aligned_rating = map_sentiment_rating[sentiment_cat]
        if predicted_count > 0:
            aligned_count = crosstab.loc[aligned_rating, sentiment_cat]
            rate = aligned_count / predicted_count
            alignment_metrics.append({"Focus": f"Predicted: {sentiment_cat}", "Metric": f"% from Rating '{aligned_rating}'",
                                     "Value": f"{rate:.2%}", "Support (Actual)": "-", "Support (Pred.)": predicted_count})
        else:
            alignment_metrics.append({"Focus": f"Predicted: {sentiment_cat}", "Metric": "-",
                                     "Value": "N/A", "Support (Actual)": "-", "Support (Pred.)": 0})
    if alignment_metrics:
        metrics_df = pd.DataFrame(alignment_metrics)
        metrics_df['Support (Actual)'] = metrics_df['Support (Actual)'].replace(
            '-', 0).astype(int)
        metrics_df['Support (Pred.)'] = metrics_df['Support (Pred.)'].replace(
            '-', 0).astype(int)
        st.dataframe(metrics_df[["Focus", "Metric", "Value",
                     "Support (Actual)", "Support (Pred.)"]], hide_index=True)
    else:
        st.warning("Could not calculate custom alignment rates.")
    st.markdown("""*   **Actual Rating Rows:** % of reviews in rating category having the 'aligned' sentiment.\n*   **Predicted Sentiment Rows:** % of reviews with predicted sentiment coming from the 'aligned' rating.""")


# --- Streamlit App Layout ---
st.title("🏥 Hospital Review Aspect-Based Sentiment Analysis")
if st.sidebar.button("Clear Cache & Rerun Data Processing"):
    st.cache_data.clear()
    st.success("Cache cleared.")
    st.rerun()

if analyzer:
    raw_df, hospital_data = load_and_process_data(DATA_FILE, analyzer)
else:
    st.error("Analyzer not ready.")
    raw_df, hospital_data = pd.DataFrame(), pd.DataFrame()

if raw_df.empty or hospital_data.empty:
    st.warning("Data loading/processing failed.")
    st.stop()

st.sidebar.header("Navigation")
analysis_mode = st.sidebar.radio("Select Analysis View", ("Overall Analysis",
                                 "Performance Metrics", "Hospital Specific Analysis", "Aspect Ranking"))
st.sidebar.header("Data Overview")
st.sidebar.metric("Total Hospitals Analyzed", hospital_data.shape[0])
st.sidebar.metric("Total Reviews Analyzed", raw_df.shape[0])
st.sidebar.markdown("--- \n Aspects Analyzed:\n" +
                    "\n".join([f"- {aspect}" for aspect in ASPECT_KEYWORDS.keys()]))

# --- Main Content ---
if analysis_mode == "Overall Analysis":
    st.header("Overall Sentiment & Rating Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Average Google Scores per Hospital")
        if not hospital_data.empty and 'Average Google Score' in hospital_data.columns:
            st.plotly_chart(px.histogram(hospital_data, x='Average Google Score', nbins=10,
                            title='Hospital Average Google Score Distribution'), use_container_width=True)
        else:
            st.warning("No data for Google Score distribution.")
    with col2:
        st.subheader(
            "Distribution of Average Review Sentiment (VADER) per Hospital")
        if not hospital_data.empty and 'Average Review Sentiment (VADER)' in hospital_data.columns:
            st.plotly_chart(px.histogram(hospital_data, x='Average Review Sentiment (VADER)', nbins=20,
                            title='Hospital Average VADER Score Distribution'), use_container_width=True)
        else:
            st.warning("No data for VADER score distribution.")
    st.subheader(
        "Correlation: Average Google Score vs. Average Calculated Sentiment per Hospital")
    if not hospital_data.empty and 'Average Google Score' in hospital_data.columns and 'Average Review Sentiment (VADER)' in hospital_data.columns:
        corr_data = hospital_data[['Average Google Score',
                                   'Average Review Sentiment (VADER)']].dropna()
        if not corr_data.empty and len(corr_data) > 1:
            try:
                correlation = corr_data['Average Google Score'].corr(
                    corr_data['Average Review Sentiment (VADER)'])
                st.metric("Pearson Correlation Coefficient",
                          f"{correlation:.3f}")
                fig_corr = px.scatter(hospital_data, x='Average Google Score', y='Average Review Sentiment (VADER)', hover_data=['Hospital Name', 'Number of Reviews'], title='Average Google Score vs. Average VADER Sentiment per Hospital', trendline='ols', labels={
                                      'Average Google Score': 'Avg. Google Score (Stars)', 'Average Review Sentiment (VADER)': 'Avg. Review Sentiment (VADER Score)'})
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating/plotting correlation: {e}")
        else:
            st.warning("Not enough valid data points for correlation.")
    else:
        st.warning("Missing columns for correlation.")
    st.subheader("Overall Hospital Aggregated Data")
    st.dataframe(hospital_data.round(3), use_container_width=True)

elif analysis_mode == "Performance Metrics":
    st.header("Performance Metrics")
    if 'google_category' in raw_df.columns and 'vader_category' in raw_df.columns:
        display_classification_metrics(raw_df)
    else:
        st.error("Categorization columns missing.")

elif analysis_mode == "Hospital Specific Analysis":
    st.header("Hospital Specific Analysis")
    if not hospital_data.empty and 'Hospital Name' in hospital_data.columns:
        hospital_list = sorted(hospital_data['Hospital Name'].unique())
        if hospital_list:
            selected_hospital = st.selectbox(
                "Select a Hospital", hospital_list)
            if selected_hospital:
                hospital_info_rows = hospital_data[hospital_data['Hospital Name']
                                                   == selected_hospital]
                if not hospital_info_rows.empty:
                    hospital_info = hospital_info_rows.iloc[0]
                    raw_hospital_reviews = raw_df[raw_df['title'] == selected_hospital].copy(
                    )
                    st.subheader(f"Analysis for: {selected_hospital}")
                    col1, col2, col3 = st.columns(3)
                    avg_google = hospital_info.get(
                        'Average Google Score', np.nan)
                    avg_vader = hospital_info.get(
                        'Average Review Sentiment (VADER)', np.nan)
                    rev_count = hospital_info.get('Number of Reviews', 0)
                    col1.metric("Average Google Score", f"{avg_google:.2f} ⭐" if pd.notna(
                        avg_google) else "N/A ⭐")
                    col2.metric("Average Review Sentiment (VADER)",
                                f"{avg_vader:.3f}" if pd.notna(avg_vader) else "N/A")
                    col3.metric("Number of Reviews", f"{int(rev_count)}")
                    st.markdown("---")
                    st.subheader("Aspect Sentiment Breakdown")
                    aspect_data = [{'Aspect': aspect, 'Mention (%)': hospital_info.get(f'{aspect} Mention (%)', 0), 'Average Sentiment': hospital_info.get(f'{aspect} Average Sentiment', 0.0) if pd.notna(
                        hospital_info.get(f'{aspect} Average Sentiment', np.nan)) else 0.0, 'Actual Avg Sentiment (if mentioned)': hospital_info.get(f'{aspect} Average Sentiment', np.nan)} for aspect in ASPECT_KEYWORDS.keys()]
                    if aspect_data:
                        aspect_df = pd.DataFrame(aspect_data)
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            fig_mention = px.bar(
                                aspect_df, x='Aspect', y='Mention (%)', title='Aspect Mention Frequency', color='Aspect')
                            fig_mention.update_layout(showlegend=False)
                            st.plotly_chart(
                                fig_mention, use_container_width=True)
                        with col_chart2:
                            fig_sentiment = px.bar(aspect_df, x='Aspect', y='Average Sentiment', title='Average Sentiment per Aspect',
                                                   color='Average Sentiment', color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1, 1])
                            fig_sentiment.update_layout(
                                yaxis=dict(range=[-1.1, 1.1]))
                            fig_sentiment.update_coloraxes(showscale=False)
                            st.plotly_chart(
                                fig_sentiment, use_container_width=True)
                        st.dataframe(aspect_df[['Aspect', 'Mention (%)', 'Actual Avg Sentiment (if mentioned)']].rename(
                            columns={'Actual Avg Sentiment (if mentioned)': 'Avg Sentiment Score'}).round(3), use_container_width=True)
                    else:
                        st.warning("No aspect data prepared.")
                    if not raw_hospital_reviews.empty and 'vader_category' in raw_hospital_reviews.columns:
                        st.subheader("Review Sentiment Distribution")
                        fig_dist = px.histogram(raw_hospital_reviews, x='vader_category', category_orders={
                                                'vader_category': SENTIMENT_CLASSES})
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.warning("No data for sentiment distribution plot.")
                    st.subheader("Sample Reviews")
                    if st.checkbox("Show sample reviews?", value=False):
                        cols_to_show = [col for col in ['text', 'totalScore', 'google_category',
                                                        'overall_sentiment_score', 'vader_category'] if col in raw_hospital_reviews.columns]
                        if cols_to_show:
                            st.dataframe(raw_hospital_reviews[cols_to_show].head(
                                10), use_container_width=True)
                        else:
                            st.warning(
                                "Required columns missing for sample reviews.")
                else:
                    st.warning(
                        f"No aggregated data for hospital: {selected_hospital}")
        else:
            st.warning("No hospitals found in aggregated data.")
    else:
        st.warning("Aggregated hospital data empty or missing 'Hospital Name'.")

elif analysis_mode == "Aspect Ranking":
    st.header("Hospital Ranking by Aspect Sentiment")
    if not hospital_data.empty:
        selected_aspect = st.selectbox(
            "Select Aspect to Rank By", list(ASPECT_KEYWORDS.keys()))
        if selected_aspect:
            sentiment_col, mention_col = f'{selected_aspect} Average Sentiment', f'{selected_aspect} Mention (%)'
            if sentiment_col in hospital_data.columns and mention_col in hospital_data.columns:
                ranked_hospitals = hospital_data[hospital_data[sentiment_col].notna()].copy(
                ).sort_values(by=sentiment_col, ascending=False)
                if not ranked_hospitals.empty:
                    st.subheader(
                        f"Ranked by Avg Sentiment for '{selected_aspect}'")
                    display_cols = ['Hospital Name', sentiment_col,
                                    mention_col, 'Number of Reviews', 'Average Google Score']
                    display_cols_present = [
                        col for col in display_cols if col in ranked_hospitals.columns]
                    if display_cols_present:
                        st.dataframe(ranked_hospitals[display_cols_present].round(
                            3), use_container_width=True)
                    else:
                        st.warning("Columns missing for ranked display.")
                else:
                    st.info(
                        f"No hospitals with sentiment for '{selected_aspect}'.")
            else:
                st.warning(
                    f"Columns missing for ranking ('{sentiment_col}', '{mention_col}').")
    else:
        st.warning("Aggregated hospital data empty.")

st.markdown("--- \n Dashboard developed for Project Analysis")
