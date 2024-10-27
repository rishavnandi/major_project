import pandas as pd
import nltk
# Download required resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt_tab")
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results"""

    textblob: str
    vader: str
    bert: str


class AspectDictionary:
    """Manages aspect-related keywords and operations"""

    ASPECTS = {
        "staff": [
            "staff",
            "doctor",
            "nurse",
            "receptionist",
            "physician",
            "specialist",
            "attendant",
            "caretaker",
            "surgeon",
            "therapist",
            "clinician",
            "technician",
            "assistant",
            "medical team",
        ],
        "cleanliness": [
            "clean",
            "dirty",
            "hygiene",
            "sanitary",
            "sanitation",
            "neat",
            "messy",
            "filthy",
            "tidy",
            "spotless",
            "dusty",
            "orderly",
            "sterile",
            "disinfection",
            "germs",
        ],
        "wait_time": [
            "wait",
            "time",
            "delay",
            "long",
            "quick",
            "queue",
            "waiting",
            "hours",
            "minutes",
            "fast",
            "slow",
            "prompt",
            "timely",
            "lag",
            "late",
            "speed",
            "duration",
            "hold",
        ],
        "facilities": [
            "facility",
            "room",
            "equipment",
            "bed",
            "resources",
            "infrastructure",
            "building",
            "furniture",
            "amenities",
            "technology",
            "device",
            "tools",
            "environment",
            "setup",
            "labs",
            "cafeteria",
            "restroom",
            "parking",
            "accessibility",
        ],
        "cost": [
            "cost",
            "price",
            "expensive",
            "affordable",
            "billing",
            "charges",
            "insurance",
            "payment",
            "fees",
            "overpriced",
            "inexpensive",
            "discount",
            "expense",
            "rates",
            "coverage",
            "deductible",
        ],
        "communication": [
            "communication",
            "information",
            "explain",
            "informed",
            "clarity",
            "questions",
            "answers",
            "understand",
            "details",
            "update",
            "report",
            "feedback",
            "interaction",
            "discussion",
            "notify",
            "guidance",
            "instructions",
        ],
    }

    @classmethod
    def get_aspects(cls) -> Dict[str, List[str]]:
        return cls.ASPECTS


class SentimentAnalyzer:
    """Handles multiple sentiment analysis methods"""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
        self._download_nltk_data()

    @staticmethod
    def _download_nltk_data():
        """Download required NLTK data"""
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")

    @staticmethod
    def _analyze_textblob(text: str) -> str:
        """Analyze sentiment using TextBlob"""
        polarity = TextBlob(text).sentiment.polarity
        return (
            "Positive"
            if polarity > 0.1
            else "Negative" if polarity < -0.1 else "Neutral"
        )

    def _analyze_vader(self, text: str) -> str:
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        return (
            "Positive"
            if scores["compound"] > 0.05
            else "Negative" if scores["compound"] < -0.05 else "Neutral"
        )

    def _analyze_bert(self, text: str) -> str:
        """Analyze sentiment using BERT"""
        result = self.bert_analyzer(text[:512])[0]
        label = int(result["label"].split()[0])
        if label <= 2:
            return "Negative"
        elif label == 3:
            return "Neutral"
        return "Positive"

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze text using all sentiment analysis methods"""
        return SentimentResult(
            textblob=self._analyze_textblob(text),
            vader=self._analyze_vader(text),
            bert=self._analyze_bert(text),
        )


class ReviewAnalyzer:
    """Handles the analysis of hospital reviews"""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_dict = AspectDictionary.get_aspects()

    @staticmethod
    def load_data(filepath: Path) -> pd.DataFrame:
        """Load and preprocess the review data"""
        try:
            df = pd.read_csv(filepath)
            required_columns = ["title", "text", "totalScore"]

            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            df = df.dropna(subset=required_columns)
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

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
                results.append(
                    {
                        "hospital": row["title"],
                        "totalScore": row["totalScore"],
                        "aspect": aspect,
                        "textblob_sentiment": sentiments.textblob,
                        "vader_sentiment": sentiments.vader,
                        "bert_sentiment": sentiments.bert,
                    }
                )

            progress_bar.progress((idx + 1) / total_rows)

        return pd.DataFrame(results)


class AspectAnalyzer:
    """Handles aspect-specific analysis and metrics"""

    @staticmethod
    def calculate_aspect_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various metrics for each aspect by hospital"""
        metrics = []

        for hospital in df["hospital"].unique():
            hospital_data = df[df["hospital"] == hospital]

            for aspect in hospital_data["aspect"].unique():
                aspect_data = hospital_data[hospital_data["aspect"] == aspect]

                # Calculate sentiment percentages
                total_mentions = len(aspect_data)
                positive_pct = (
                    aspect_data["bert_sentiment"] == "Positive"
                ).mean() * 100
                negative_pct = (
                    aspect_data["bert_sentiment"] == "Negative"
                ).mean() * 100
                neutral_pct = (aspect_data["bert_sentiment"] == "Neutral").mean() * 100

                # Calculate average score for this aspect
                avg_score = aspect_data["totalScore"].mean()

                metrics.append(
                    {
                        "hospital": hospital,
                        "aspect": aspect,
                        "mentions": total_mentions,
                        "positive_pct": positive_pct,
                        "negative_pct": negative_pct,
                        "neutral_pct": neutral_pct,
                        "avg_score": avg_score,
                    }
                )

        return pd.DataFrame(metrics)


class DashboardVisualizer:
    """Handles the visualization of analysis results"""

    @staticmethod
    def plot_sentiment_distribution(aspect_df: pd.DataFrame, title: str) -> None:
        """Plot sentiment distribution across different models"""
        fig, axs = plt.subplots(1, 3, figsize=(25, 8))

        sns.countplot(
            data=aspect_df,
            x="aspect",
            hue="textblob_sentiment",
            ax=axs[0],
            palette="viridis",
        )
        axs[0].set_title("TextBlob Sentiment")

        sns.countplot(
            data=aspect_df,
            x="aspect",
            hue="vader_sentiment",
            ax=axs[1],
            palette="magma",
        )
        axs[1].set_title("VADER Sentiment")

        sns.countplot(
            data=aspect_df,
            x="aspect",
            hue="bert_sentiment",
            ax=axs[2],
            palette="coolwarm",
        )
        axs[2].set_title("BERT Sentiment")

        plt.suptitle(title)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_aspect_comparison(
        metrics_df: pd.DataFrame, aspect: str, sort_by: str
    ) -> None:
        """Plot comparison of hospitals for a specific aspect"""
        aspect_data = metrics_df[metrics_df["aspect"] == aspect].sort_values(
            sort_by, ascending=False
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 24))

        # Plot sentiment distribution
        sentiment_data = pd.melt(
            aspect_data,
            id_vars=["hospital"],
            value_vars=["positive_pct", "neutral_pct", "negative_pct"],
            var_name="sentiment",
            value_name="percentage",
        )

        sns.barplot(
            data=sentiment_data,
            x="hospital",
            y="percentage",
            hue="sentiment",
            ax=ax1,
            palette="RdYlBu",
        )
        ax1.set_title(f"Sentiment Distribution for {aspect}")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

        # Plot mention counts
        sns.barplot(
            data=aspect_data, x="hospital", y="mentions", ax=ax2, palette="viridis"
        )
        ax2.set_title(f"Number of Mentions for {aspect}")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        st.pyplot(fig)


def main():
    """Main application function"""
    st.title("Aspect-Based Sentiment Analysis of Hospital Reviews In Bhubaneswar")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        return

    try:
        analyzer = ReviewAnalyzer()
        df = analyzer.load_data(uploaded_file)

        st.write("Data Sample:", df.head())

        with st.spinner("Analyzing reviews..."):
            aspect_df = analyzer.analyze_reviews(df)
            aspect_metrics = AspectAnalyzer.calculate_aspect_metrics(aspect_df)

        visualizer = DashboardVisualizer()

        # Overall sentiment distribution
        st.subheader("Overall Sentiment Distribution by Aspect")
        visualizer.plot_sentiment_distribution(
            aspect_df, "Sentiment Distribution Across All Hospitals"
        )

        # Aspect-specific analysis
        st.subheader("Aspect-Specific Analysis")

        col1, col2 = st.columns(2)

        with col1:
            selected_aspect = st.selectbox(
                "Select an aspect to analyze", sorted(aspect_df["aspect"].unique())
            )

        with col2:
            sort_metric = st.selectbox(
                "Sort hospitals by",
                [
                    "positive_pct",
                    "negative_pct",
                    "neutral_pct",
                    "mentions",
                    "avg_score",
                ],
                format_func=lambda x: {
                    "positive_pct": "Positive Sentiment %",
                    "negative_pct": "Negative Sentiment %",
                    "neutral_pct": "Neutral Sentiment %",
                    "mentions": "Number of Mentions",
                    "avg_score": "Average Score",
                }[x],
            )

        visualizer.plot_aspect_comparison(aspect_metrics, selected_aspect, sort_metric)

        # Detailed metrics table
        st.subheader("Detailed Metrics by Hospital and Aspect")

        metrics_view = aspect_metrics.copy()
        metrics_view["positive_pct"] = (
            metrics_view["positive_pct"].round(1).astype(str) + "%"
        )
        metrics_view["negative_pct"] = (
            metrics_view["negative_pct"].round(1).astype(str) + "%"
        )
        metrics_view["neutral_pct"] = (
            metrics_view["neutral_pct"].round(1).astype(str) + "%"
        )
        metrics_view["avg_score"] = metrics_view["avg_score"].round(2)

        st.dataframe(
            metrics_view.sort_values([sort_metric], ascending=False),
            use_container_width=True,
        )

        # Average ratings
        st.subheader("Overall Hospital Ratings")
        avg_rating = (
            df.groupby("title")["totalScore"]
            .agg(["mean", "count", "std"])
            .round(2)
            .reset_index()
        )
        avg_rating.columns = [
            "Hospital",
            "Average Rating",
            "Number of Reviews",
            "Standard Deviation",
        ]
        st.table(avg_rating)

        # Hospital-specific analysis
        st.subheader("Hospital-Specific Analysis")
        selected_hospital = st.selectbox("Select a hospital", df["title"].unique())

        hospital_data = aspect_df[aspect_df["hospital"] == selected_hospital]
        visualizer.plot_sentiment_distribution(
            hospital_data, f"Sentiment Distribution for {selected_hospital}"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
