from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.figure_factory as ff
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK resources only once
NLTK_RESOURCES = ["vader_lexicon", "stopwords",
                  "wordnet", "punkt", "averaged_perceptron_tagger"]
for resource in NLTK_RESOURCES:
    nltk.download(resource, quiet=True)


@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results with confidence scores"""
    textblob: str
    vader: str
    bert: str
    textblob_score: float
    vader_score: float
    bert_score: float
    confidence: float


class DataNormalizer:
    """Handles data normalization and preprocessing"""

    def __init__(self):
        self.score_scaler = MinMaxScaler(feature_range=(1, 5))

    def normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical scores to a 1-5 range"""
        score_columns = ['textblob_score',
                         'vader_score', 'bert_score', 'totalScore']
        df_normalized = df.copy()

        for col in [c for c in score_columns if c in df.columns]:
            shaped_data = df[col].values.reshape(-1, 1)
            df_normalized[f'{col}_normalized'] = self.score_scaler.fit_transform(
                shaped_data)

        return df_normalized

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text data by removing special characters and standardizing format"""
        # Convert to lowercase and remove special characters
        text = ''.join(char for char in text.lower()
                       if char.isalnum() or char in ' .,!?')
        # Remove extra whitespace
        return ' '.join(text.split())


class AspectDictionary:
    """Aspect dictionary with weighted keywords and hierarchical categorization"""
    ASPECTS = {
        "staff": {
            "keywords": {
                "primary": ["staff", "doctor", "nurse", "receptionist", "physician"],
                "secondary": ["specialist", "attendant", "caretaker", "surgeon", "therapist"],
                "related": ["team", "personnel", "medical team", "healthcare provider"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        },
        "cleanliness": {
            "keywords": {
                "primary": ["clean", "dirty", "hygiene", "sanitary", "sanitation"],
                "secondary": ["neat", "messy", "filthy", "tidy", "spotless"],
                "related": ["dusty", "maintained", "sterile", "pristine"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        },
        "wait_time": {
            "keywords": {
                "primary": ["wait", "waiting", "queue", "delay"],
                "secondary": ["long", "quick", "fast", "slow"],
                "related": ["hours", "minutes", "prompt", "time"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        },
        "facilities": {
            "keywords": {
                "primary": ["facility", "room", "equipment", "building"],
                "secondary": ["bed", "amenities", "technology", "infrastructure"],
                "related": ["parking", "restroom", "modern", "outdated"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        },
        "cost": {
            "keywords": {
                "primary": ["cost", "price", "expensive", "affordable"],
                "secondary": ["billing", "charges", "insurance", "payment"],
                "related": ["fees", "overpriced", "reasonable", "worth"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        },
        "communication": {
            "keywords": {
                "primary": ["communication", "explain", "information", "informed"],
                "secondary": ["clarity", "questions", "answers", "understand"],
                "related": ["update", "responsive", "clear", "helpful"]
            },
            "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}
        }
    }

    @classmethod
    def get_aspects(cls) -> Dict[str, Dict]:
        return cls.ASPECTS


class SentimentAnalyzer:
    """Sentiment analyzer with confidence scoring and ensemble methods"""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = pipeline("sentiment-analysis",
                                      model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.normalizer = DataNormalizer()

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on agreement between models"""
        # Higher agreement = higher confidence
        return float(1 / (1 + np.std(scores)))

    def _analyze_textblob(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using TextBlob"""
        polarity = TextBlob(text).sentiment.polarity
        score = 2.5 + (polarity * 2.5)
        sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
        return sentiment, score

    def _analyze_vader(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using VADER"""
        compound = self.vader_analyzer.polarity_scores(text)['compound']
        score = 2.5 + (compound * 2.5)
        sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
        return sentiment, score

    def _analyze_bert(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using BERT"""
        result = self.bert_analyzer(text[:512])[0]
        label = int(result["label"].split()[0])
        sentiment = "Negative" if label <= 2 else "Neutral" if label == 3 else "Positive"
        return sentiment, float(label)

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze text using all sentiment analysis methods with confidence scoring"""
        normalized_text = self.normalizer.normalize_text(text)

        textblob_sentiment, textblob_score = self._analyze_textblob(
            normalized_text)
        vader_sentiment, vader_score = self._analyze_vader(normalized_text)
        bert_sentiment, bert_score = self._analyze_bert(normalized_text)

        confidence = self._calculate_confidence(
            [textblob_score, vader_score, bert_score])

        return SentimentResult(
            textblob=textblob_sentiment,
            vader=vader_sentiment,
            bert=bert_sentiment,
            textblob_score=textblob_score,
            vader_score=vader_score,
            bert_score=bert_score,
            confidence=confidence
        )


class ReviewAnalyzer:
    """Review analyzer with aspect extraction and sentiment analysis"""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_dict = AspectDictionary.get_aspects()
        self.normalizer = DataNormalizer()

    def extract_aspects(self, text: str) -> Dict[str, float]:
        """Extract aspects with confidence scores based on keyword weights"""
        aspects = {}
        normalized_text = self.normalizer.normalize_text(text)
        # Using set for faster lookups
        tokens = set(nltk.word_tokenize(normalized_text))

        for aspect, aspect_data in self.aspect_dict.items():
            score = 0
            hits = 0

            for category, keywords in aspect_data["keywords"].items():
                weight = aspect_data["weights"][category]
                for keyword in keywords:
                    if keyword in normalized_text:  # Simple substring match is faster
                        score += weight
                        hits += 1

            if hits > 0:
                aspects[aspect] = score / hits

        return aspects

    def analyze_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze aspects and sentiments in reviews"""
        results = []
        total_rows = len(df)
        progress_bar = st.progress(0)

        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                # Extract aspects and analyze sentiment
                aspects = self.extract_aspects(row["text"])
                sentiments = self.sentiment_analyzer.analyze_text(row["text"])

                # Add results for each aspect
                for aspect, confidence in aspects.items():
                    results.append({
                        "hospital": row["title"],
                        "totalScore": row["totalScore"],
                        "aspect": aspect,
                        "aspect_confidence": confidence,
                        "textblob_sentiment": sentiments.textblob,
                        "vader_sentiment": sentiments.vader,
                        "bert_sentiment": sentiments.bert,
                        "textblob_score": sentiments.textblob_score,
                        "vader_score": sentiments.vader_score,
                        "bert_score": sentiments.bert_score,
                        "sentiment_confidence": sentiments.confidence
                    })

                progress_bar.progress((idx + 1) / total_rows)

            except Exception as e:
                logger.error(f"Error processing review {idx}: {str(e)}")
                continue

        # Create DataFrame and normalize scores
        return self.normalizer.normalize_scores(pd.DataFrame(results))


class AnalysisUtils:
    """Utility class for analysis methods"""

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

            metrics[model] = {
                'mse': mean_squared_error(df['totalScore'], df[score_col]),
                'tolerance_accuracy': np.mean(np.abs(df['totalScore'] - df[score_col]) <= 1.0),
                'correlation': df['totalScore'].corr(df[score_col])
            }

        return metrics

    @staticmethod
    def score_to_category(score: float) -> str:
        """Convert numerical score to sentiment category"""
        if score <= 2:
            return "Negative"
        elif score <= 3.5:
            return "Neutral"
        else:
            return "Positive"

    @staticmethod
    def calculate_confusion_matrices(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Calculate confusion matrices for each model"""
        actual_categories = df['totalScore'].apply(
            AnalysisUtils.score_to_category)
        categories = ['Negative', 'Neutral', 'Positive']
        matrices = {}

        for model in ['textblob', 'vader', 'bert']:
            predicted_categories = df[f'{model}_sentiment']
            cm = confusion_matrix(
                actual_categories, predicted_categories, labels=categories)
            matrices[model] = cm.astype(
                'float') / cm.sum(axis=1)[:, np.newaxis]

        return matrices, categories


class Visualizer:
    """Handles visualization of analysis results"""

    @staticmethod
    def plot_accuracy_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot accuracy metrics comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        models = list(metrics.keys())
        metrics_to_plot = {
            'mse': [metrics[m]['mse'] for m in models],
            'tolerance_accuracy': [metrics[m]['tolerance_accuracy'] * 100 for m in models],
            'correlation': [metrics[m]['correlation'] for m in models]
        }

        titles = ['Mean Squared Error',
                  'Accuracy (±1 point tolerance)', 'Correlation with Total Score']
        ylabels = ['MSE', 'Accuracy (%)', 'Correlation Coefficient']

        for i, (metric, values) in enumerate(metrics_to_plot.items()):
            sns.barplot(x=models, y=values, ax=axes[i], palette='viridis')
            axes[i].set_title(titles[i])
            axes[i].set_ylabel(ylabels[i])

        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_score_comparison(df: pd.DataFrame) -> None:
        """Plot comparison of predicted scores vs actual scores"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, model in enumerate(['textblob', 'vader', 'bert']):
            score_col = f'{model}_score'

            sns.scatterplot(data=df, x='totalScore',
                            y=score_col, ax=axes[idx], alpha=0.5)
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

        models = {
            'textblob_sentiment': {'ax': axs[0], 'palette': 'viridis', 'title': 'TextBlob Sentiment'},
            'vader_sentiment': {'ax': axs[1], 'palette': 'magma', 'title': 'VADER Sentiment'},
            'bert_sentiment': {'ax': axs[2], 'palette': 'coolwarm', 'title': 'BERT Sentiment'}
        }

        for model, config in models.items():
            sns.countplot(data=aspect_df, x="aspect", hue=model,
                          ax=config['ax'], palette=config['palette'])
            config['ax'].set_title(config['title'])
            config['ax'].tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_confusion_matrices(confusion_matrices: Dict[str, np.ndarray], categories: List[str]) -> None:
        """Plot confusion matrices using plotly"""
        cols = st.columns(3)

        for idx, (model, cm) in enumerate(confusion_matrices.items()):
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=categories,
                y=categories,
                annotation_text=np.around(cm, decimals=2),
                colorscale='Viridis'
            )

            fig.update_layout(
                title=f"{model.upper()} Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=400,
                height=400,
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )

            with cols[idx]:
                st.plotly_chart(fig)


def main():
    """Main application function"""
    st.title("Hospital Review Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis")
        return

    try:
        # Load and analyze data
        df = pd.read_csv(uploaded_file)

        with st.spinner("Analyzing reviews..."):
            # Run analysis pipeline
            analyzer = ReviewAnalyzer()
            aspect_df = analyzer.analyze_reviews(df)

            # Calculate metrics
            aspect_metrics = AnalysisUtils.calculate_aspect_metrics(aspect_df)
            accuracy_metrics = AnalysisUtils.calculate_accuracy_metrics(
                aspect_df)
            confusion_matrices, categories = AnalysisUtils.calculate_confusion_matrices(
                aspect_df)

            # Confusion Matrix Analysis
            st.header("Confusion Matrix Analysis")
            st.write("""
            These confusion matrices show how well each model's sentiment predictions align with 
            the categorized actual ratings. The values show the proportion of each actual category 
            that was predicted as each sentiment category.
            """)

            Visualizer.plot_confusion_matrices(confusion_matrices, categories)

            # Model Performance Summary
            st.subheader("Model Performance Summary")
            model_accuracy = {model: np.trace(
                cm) / np.sum(cm) for model, cm in confusion_matrices.items()}
            summary_df = pd.DataFrame({
                'Model': model_accuracy.keys(),
                'Overall Accuracy': [f"{acc:.2%}" for acc in model_accuracy.values()]
            })
            st.table(summary_df)

            # Accuracy Metrics Section
            st.header("Model Accuracy Analysis")
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
            Visualizer.plot_accuracy_metrics(accuracy_metrics)

            # Score comparison plots
            st.subheader("Predicted vs Actual Scores")
            Visualizer.plot_score_comparison(aspect_df)

            # Sentiment Distribution
            st.header("Sentiment Analysis")
            st.subheader("Overall Sentiment Distribution by Aspect")
            Visualizer.plot_sentiment_distribution(
                aspect_df, "Sentiment Distribution Across All Hospitals")

            # Aspect-specific analysis
            st.header("Aspect-Specific Analysis")

            col1, col2 = st.columns(2)
            with col1:
                selected_aspect = st.selectbox(
                    "Select aspect", sorted(aspect_df["aspect"].unique()))
            with col2:
                sort_options = {
                    "positive_pct": "Positive Sentiment %",
                    "negative_pct": "Negative Sentiment %",
                    "neutral_pct": "Neutral Sentiment %",
                    "mentions": "Number of Mentions",
                    "avg_score": "Average Score"
                }
                sort_metric = st.selectbox("Sort by", list(sort_options.keys()),
                                           format_func=lambda x: sort_options[x])

            # Display detailed metrics
            st.subheader("Detailed Metrics by Hospital and Aspect")
            metrics_view = aspect_metrics.copy()

            # Format percentage columns
            for col in ['positive_pct', 'negative_pct', 'neutral_pct']:
                metrics_view[col] = metrics_view[col].round(
                    1).astype(str) + "%"

            # Round score columns
            metrics_view[['avg_score', 'bert_avg', 'textblob_avg', 'vader_avg']] = \
                metrics_view[['avg_score', 'bert_avg',
                              'textblob_avg', 'vader_avg']].round(2)

            st.dataframe(metrics_view.sort_values(
                [sort_metric], ascending=False), use_container_width=True)

            # Hospital-specific analysis
            st.header("Hospital-Specific Analysis")
            selected_hospital = st.selectbox(
                "Select a hospital", df["title"].unique())

            # Filter data for selected hospital
            hospital_data = aspect_df[aspect_df["hospital"]
                                      == selected_hospital]

            # Display hospital-specific metrics
            st.subheader(f"Sentiment Distribution for {selected_hospital}")
            Visualizer.plot_sentiment_distribution(
                hospital_data, f"Sentiment Distribution for {selected_hospital}")

            # Model performance comparison for selected hospital
            st.subheader("Model Performance Comparison")
            hospital_metrics = {
                model: {
                    'mse': mean_squared_error(hospital_data['totalScore'], hospital_data[f'{model}_score']),
                    'tolerance_accuracy': np.mean(
                        np.abs(hospital_data['totalScore'] -
                               hospital_data[f'{model}_score']) <= 1.0
                    ),
                    'correlation': hospital_data['totalScore'].corr(hospital_data[f'{model}_score'])
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
            best_model = min(hospital_metrics.items(),
                             key=lambda x: x[1]['mse'])[0].upper()
            most_mentioned = hospital_data['aspect'].value_counts(
            ).idxmax() if not hospital_data.empty else "N/A"

            summary_stats = pd.DataFrame({
                'Metric': ['Total Reviews', 'Average Rating', 'Most Mentioned Aspect', 'Best Performing Model'],
                'Value': [
                    len(hospital_data),
                    f"{hospital_data['totalScore'].mean(
                    ):.2f}" if not hospital_data.empty else "N/A",
                    most_mentioned,
                    best_model
                ]
            })

            st.table(summary_stats)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
