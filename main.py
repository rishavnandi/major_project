from sklearn.metrics import mean_squared_error
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import pandas as pd
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results"""
    textblob: str
    vader: str
    bert: str
    textblob_score: float
    vader_score: float
    bert_score: float


class AspectDictionary:
    """Manages aspect-related keywords and operations"""
    ASPECTS = {
        "staff": ["staff", "doctor", "nurse", "receptionist", "physician", "specialist",
                  "attendant", "caretaker", "surgeon", "therapist", "clinician"],
        "cleanliness": ["clean", "dirty", "hygiene", "sanitary", "sanitation", "neat",
                        "messy", "filthy", "tidy", "spotless", "dusty"],
        "wait_time": ["wait", "time", "delay", "long", "quick", "queue", "waiting",
                      "hours", "minutes", "fast", "slow", "prompt"],
        "facilities": ["facility", "room", "equipment", "bed", "building", "amenities",
                       "technology", "infrastructure", "parking", "restroom"],
        "cost": ["cost", "price", "expensive", "affordable", "billing", "charges",
                 "insurance", "payment", "fees", "overpriced"],
        "communication": ["communication", "information", "explain", "informed", "clarity",
                          "questions", "answers", "understand", "update"]
    }

    @classmethod
    def get_aspects(cls) -> Dict[str, List[str]]:
        return cls.ASPECTS


class SentimentAnalyzer:
    """Handles multiple sentiment analysis methods"""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = pipeline("sentiment-analysis",
                                      model="nlptown/bert-base-multilingual-uncased-sentiment")

    def _convert_score_to_5_scale(self, score: float, model: str) -> float:
        """Convert model-specific scores to 5-point scale"""
        if model in ['textblob', 'vader']:  # -1 to 1 scale
            return 2.5 + (score * 2.5)
        elif model == 'bert':     # 1 to 5 scale
            return float(score)
        return score

    def _analyze_textblob(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using TextBlob"""
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        score = self._convert_score_to_5_scale(polarity, 'textblob')
        sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
        return sentiment, score

    def _analyze_vader(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        score = self._convert_score_to_5_scale(compound, 'vader')
        sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
        return sentiment, score

    def _analyze_bert(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using BERT"""
        result = self.bert_analyzer(text[:512])[0]
        label = int(result["label"].split()[0])
        score = float(label)

        if label <= 2:
            sentiment = "Negative"
        elif label == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        return sentiment, score

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze text using all sentiment analysis methods"""
        textblob_sentiment, textblob_score = self._analyze_textblob(text)
        vader_sentiment, vader_score = self._analyze_vader(text)
        bert_sentiment, bert_score = self._analyze_bert(text)

        return SentimentResult(
            textblob=textblob_sentiment,
            vader=vader_sentiment,
            bert=bert_sentiment,
            textblob_score=textblob_score,
            vader_score=vader_score,
            bert_score=bert_score
        )


class ReviewAnalyzer:
    """Handles the analysis of hospital reviews"""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_dict = AspectDictionary.get_aspects()

    def extract_aspects(self, text: str) -> List[str]:
        """Extract aspects from review text"""
        aspects = set()
        tokens = nltk.word_tokenize(text.lower())

        for word in tokens:
            for aspect, keywords in self.aspect_dict.items():
                if word in keywords:
                    aspects.add(aspect)
        return list(aspects)

    def analyze_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze aspects and sentiments in reviews"""
        results = []
        total_rows = len(df)
        progress_bar = st.progress(0)

        for idx, (_, row) in enumerate(df.iterrows()):
            aspects = self.extract_aspects(row["text"])
            sentiments = self.sentiment_analyzer.analyze_text(row["text"])

            for aspect in aspects:
                results.append({
                    "hospital": row["title"],
                    "totalScore": row["totalScore"],
                    "aspect": aspect,
                    "textblob_sentiment": sentiments.textblob,
                    "vader_sentiment": sentiments.vader,
                    "bert_sentiment": sentiments.bert,
                    "textblob_score": sentiments.textblob_score,
                    "vader_score": sentiments.vader_score,
                    "bert_score": sentiments.bert_score
                })

            progress_bar.progress((idx + 1) / total_rows)

        return pd.DataFrame(results)


class AspectAnalyzer:
    """Handles aspect-specific analysis and metrics"""
    @staticmethod
    def calculate_aspect_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for each aspect by hospital"""
        metrics = []

        for hospital in df["hospital"].unique():
            hospital_data = df[df["hospital"] == hospital]

            for aspect in hospital_data["aspect"].unique():
                aspect_data = hospital_data[hospital_data["aspect"] == aspect]

                metrics.append({
                    "hospital": hospital,
                    "aspect": aspect,
                    "mentions": len(aspect_data),
                    "positive_pct": (aspect_data["bert_sentiment"] == "Positive").mean() * 100,
                    "negative_pct": (aspect_data["bert_sentiment"] == "Negative").mean() * 100,
                    "neutral_pct": (aspect_data["bert_sentiment"] == "Neutral").mean() * 100,
                    "avg_score": aspect_data["totalScore"].mean(),
                    "bert_avg": aspect_data["bert_score"].mean(),
                    "textblob_avg": aspect_data["textblob_score"].mean(),
                    "vader_avg": aspect_data["vader_score"].mean()
                })

        return pd.DataFrame(metrics)

    @staticmethod
    def calculate_accuracy_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy metrics for each sentiment model"""
        metrics = {}

        for model in ['textblob', 'vader', 'bert']:
            score_col = f'{model}_score'

            mse = mean_squared_error(df['totalScore'], df[score_col])
            tolerance_accuracy = np.mean(
                np.abs(df['totalScore'] - df[score_col]) <= 1.0)
            correlation = df['totalScore'].corr(df[score_col])

            metrics[model] = {
                'mse': mse,
                'tolerance_accuracy': tolerance_accuracy,
                'correlation': correlation
            }

        return metrics


class DashboardVisualizer:
    """Handles the visualization of analysis results"""
    @staticmethod
    def plot_accuracy_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot accuracy metrics comparison"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        models = list(metrics.keys())
        mse_values = [metrics[m]['mse'] for m in models]
        accuracy_values = [
            metrics[m]['tolerance_accuracy'] * 100 for m in models]
        correlation_values = [metrics[m]['correlation'] for m in models]

        # MSE plot
        sns.barplot(x=models, y=mse_values, ax=ax1, palette='viridis')
        ax1.set_title('Mean Squared Error')
        ax1.set_ylabel('MSE')

        # Accuracy plot
        sns.barplot(x=models, y=accuracy_values, ax=ax2, palette='viridis')
        ax2.set_title('Accuracy (±1 point tolerance)')
        ax2.set_ylabel('Accuracy (%)')

        # Correlation plot
        sns.barplot(x=models, y=correlation_values, ax=ax3, palette='viridis')
        ax3.set_title('Correlation with Total Score')
        ax3.set_ylabel('Correlation Coefficient')

        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_score_comparison(df: pd.DataFrame) -> None:
        """Plot comparison of predicted scores vs actual scores"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, model in enumerate(['textblob', 'vader', 'bert']):
            score_col = f'{model}_score'

            sns.scatterplot(
                data=df,
                x='totalScore',
                y=score_col,
                ax=axes[idx],
                alpha=0.5
            )

            axes[idx].plot([1, 5], [1, 5], 'r--', alpha=0.5)
            axes[idx].set_title(f'{model.upper()} Score vs Actual Score')
            axes[idx].set_xlabel('Actual Score')
            axes[idx].set_ylabel('Predicted Score')

        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_sentiment_distribution(aspect_df: pd.DataFrame, title: str) -> None:
        """Plot sentiment distribution across different models"""
        fig, axs = plt.subplots(1, 3, figsize=(25, 8))

        sns.countplot(data=aspect_df, x="aspect",
                      hue="textblob_sentiment", ax=axs[0], palette="viridis")
        axs[0].set_title("TextBlob Sentiment")
        axs[0].tick_params(axis='x', rotation=45)

        sns.countplot(data=aspect_df, x="aspect",
                      hue="vader_sentiment", ax=axs[1], palette="magma")
        axs[1].set_title("VADER Sentiment")
        axs[1].tick_params(axis='x', rotation=45)

        sns.countplot(data=aspect_df, x="aspect",
                      hue="bert_sentiment", ax=axs[2], palette="coolwarm")
        axs[2].set_title("BERT Sentiment")
        axs[2].tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()
        st.pyplot(fig)


def main():
    """Main application function"""
    st.title("Enhanced Hospital Review Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis")
        return

    try:
        # Load and analyze data
        df = pd.read_csv(uploaded_file)
        analyzer = ReviewAnalyzer()

        with st.spinner("Analyzing reviews..."):
            aspect_df = analyzer.analyze_reviews(df)
            aspect_metrics = AspectAnalyzer.calculate_aspect_metrics(aspect_df)
            accuracy_metrics = AspectAnalyzer.calculate_accuracy_metrics(
                aspect_df)

        visualizer = DashboardVisualizer()

        # Accuracy Metrics Section
        st.header("Model Accuracy Analysis")

        # Display metrics table
        metrics_df = pd.DataFrame({
            model: {
                'Mean Squared Error': f"{metrics['mse']:.3f}",
                'Accuracy (±1 point)': f"{metrics['tolerance_accuracy']*100:.1f}%",
                'Correlation': f"{metrics['correlation']:.3f}"
            }
            for model, metrics in accuracy_metrics.items()
        }).transpose()

        st.subheader("Accuracy Metrics Comparison")
        st.table(metrics_df)

        # Plot accuracy metrics
        visualizer.plot_accuracy_metrics(accuracy_metrics)

        # Score comparison plots
        st.subheader("Predicted vs Actual Scores")
        visualizer.plot_score_comparison(aspect_df)

        # Sentiment Distribution
        st.header("Sentiment Analysis")
        st.subheader("Overall Sentiment Distribution by Aspect")
        visualizer.plot_sentiment_distribution(
            aspect_df, "Sentiment Distribution Across All Hospitals")

        # Aspect-specific analysis
        st.header("Aspect-Specific Analysis")

        col1, col2 = st.columns(2)
        with col1:
            selected_aspect = st.selectbox(
                "Select aspect", sorted(aspect_df["aspect"].unique()))
        with col2:
            sort_metric = st.selectbox(
                "Sort by",
                ["positive_pct", "negative_pct",
                    "neutral_pct", "mentions", "avg_score"],
                format_func=lambda x: {
                    "positive_pct": "Positive Sentiment %",
                    "negative_pct": "Negative Sentiment %",
                    "neutral_pct": "Neutral Sentiment %",
                    "mentions": "Number of Mentions",
                    "avg_score": "Average Score"
                }[x]
            )

        # Display detailed metrics
        st.subheader("Detailed Metrics by Hospital and Aspect")
        metrics_view = aspect_metrics.copy()
        for col in ['positive_pct', 'negative_pct', 'neutral_pct']:
            metrics_view[col] = metrics_view[col].round(1).astype(str) + "%"
        metrics_view[['avg_score', 'bert_avg', 'textblob_avg', 'vader_avg']] = \
            metrics_view[['avg_score', 'bert_avg',
                          'textblob_avg', 'vader_avg']].round(2)

        st.dataframe(
            metrics_view.sort_values([sort_metric], ascending=False),
            use_container_width=True
        )

        # Hospital-specific analysis
        st.header("Hospital-Specific Analysis")
        selected_hospital = st.selectbox(
            "Select a hospital", df["title"].unique())

        # Filter data for selected hospital
        hospital_data = aspect_df[aspect_df["hospital"] == selected_hospital]

        # Display hospital-specific metrics
        st.subheader(f"Sentiment Distribution for {selected_hospital}")
        visualizer.plot_sentiment_distribution(
            hospital_data,
            f"Sentiment Distribution for {selected_hospital}"
        )

        # Model performance comparison for selected hospital
        st.subheader("Model Performance Comparison")
        hospital_metrics = {
            model: {
                'mse': mean_squared_error(hospital_data['totalScore'],
                                          hospital_data[f'{model}_score']),
                'tolerance_accuracy': np.mean(
                    np.abs(hospital_data['totalScore'] -
                           hospital_data[f'{model}_score']) <= 1.0
                ),
                'correlation': hospital_data['totalScore'].corr(
                    hospital_data[f'{model}_score']
                )
            }
            for model in ['textblob', 'vader', 'bert']
        }

        hospital_metrics_df = pd.DataFrame({
            model: {
                'Mean Squared Error': f"{metrics['mse']:.3f}",
                'Accuracy (±1 point)': f"{metrics['tolerance_accuracy']*100:.1f}%",
                'Correlation': f"{metrics['correlation']:.3f}"
            }
            for model, metrics in hospital_metrics.items()
        }).transpose()

        st.table(hospital_metrics_df)

        # Summary statistics
        st.subheader("Summary Statistics")
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Reviews',
                'Average Rating',
                'Most Mentioned Aspect',
                'Best Performing Model'
            ],
            'Value': [
                len(hospital_data),
                f"{hospital_data['totalScore'].mean():.2f}",
                hospital_data['aspect'].mode().iloc[0],
                min(hospital_metrics.items(),
                    key=lambda x: x[1]['mse'])[0].upper()
            ]
        })
        st.table(summary_stats)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
