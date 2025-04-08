# -*- coding: utf-8 -*-
"""
Hospital Review Analysis Dashboard (Minimal Refactor v2)

Analyzes patient reviews for sentiment and aspects, focusing on core functionality,
data cleaning, and error fixes.
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, Pipeline as HFPipeline
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
# NLTK imports
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# Sentiment Analysis Libraries
# Scikit-learn imports

# --- Configuration & Constants ---

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NLTK Resource Management - Ensuring 'punkt' is listed
NLTK_RESOURCES = {
    "corpora": ["stopwords", "wordnet", "omw-1.4"],
    "tokenizers": ["punkt"],  # Explicitly ensure punkt is here
    "taggers": ["averaged_perceptron_tagger"],
    "sentiment": ["vader_lexicon"],
}

# Model Identifiers & Column Naming
TEXTBLOB, VADER, BERT = "textblob", "vader", "bert"
SENTIMENT_MODELS = [TEXTBLOB, VADER, BERT]
RAW_SCORE_SUFFIX = "_score"
SENTIMENT_SUFFIX = "_sentiment"

# Sentiment Categories
POSITIVE, NEGATIVE, NEUTRAL = "Positive", "Negative", "Neutral"
SENTIMENT_CATEGORIES = [NEGATIVE, NEUTRAL, POSITIVE]

BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MIN_REVIEW_WORDS = 3  # Filter out very short reviews

# Aspect Keywords (Simplified from Class) - Keep as is
ASPECT_KEYWORDS = {
    "staff": {"keywords": {"primary": ["staff", "doctor", "nurse", "receptionist", "physician", "provider"], "secondary": ["specialist", "attendant", "caretaker", "surgeon", "therapist", "consultant"], "related": ["team", "personnel", "medical team", "healthcare provider", "employee"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}},
    "cleanliness": {"keywords": {"primary": ["clean", "dirty", "hygiene", "sanitary", "sanitation", "sterile"], "secondary": ["neat", "messy", "filthy", "tidy", "spotless", "immaculate"], "related": ["dusty", "maintained", "pristine", "unclean"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}},
    "wait_time": {"keywords": {"primary": ["wait", "waiting", "queue", "delay", "appointment time", "schedule"], "secondary": ["long", "quick", "fast", "slow", "prompt", "punctual"], "related": ["hours", "minutes", "time", "duration", "on time"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}},
    "facilities": {"keywords": {"primary": ["facility", "room", "equipment", "building", "hospital", "clinic"], "secondary": ["bed", "amenities", "technology", "infrastructure", "space", "environment"], "related": ["parking", "restroom", "modern", "outdated", "comfortable", "layout"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}},
    "cost": {"keywords": {"primary": ["cost", "price", "expensive", "affordable", "bill", "charge"], "secondary": ["billing", "charges", "insurance", "payment", "fee", "copay"], "related": ["fees", "overpriced", "reasonable", "worth", "value", "financial"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}},
    "communication": {"keywords": {"primary": ["communication", "explain", "explanation", "information", "informed", "listen"], "secondary": ["clarity", "questions", "answers", "understand", "discuss", "consultation"], "related": ["update", "responsive", "clear", "helpful", "attentive", "instructions"]}, "weights": {"primary": 1.0, "secondary": 0.8, "related": 0.6}}
}
# --- NLTK Setup ---


def download_nltk_resources():
    """Downloads required NLTK resources if not already present."""
    logger.info("Checking NLTK resources...")
    for resource_type, resources in NLTK_RESOURCES.items():
        for resource in resources:
            try:
                # Construct the path based on type for nltk.data.find
                path_prefix = ""
                if resource_type == "corpora":
                    path_prefix = f"corpora/{resource}"
                elif resource_type == "tokenizers":
                    path_prefix = f"tokenizers/{resource}"
                elif resource_type == "taggers":
                    path_prefix = f"taggers/{resource}"
                elif resource_type == "sentiment":
                    path_prefix = f"sentiment/{resource}"
                else:
                    path_prefix = resource  # Fallback for potential other types

                if path_prefix:  # Only search if we constructed a path
                    nltk.data.find(path_prefix)
                    logger.debug(f"NLTK resource '{resource}' found.")
                else:
                    logger.warning(
                        f"Unknown NLTK resource type for '{resource}'. Skipping check.")

            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                # Consider removing quiet=True temporarily if downloads fail silently
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(
                    f"Error checking/downloading NLTK resource {resource}: {e}")
    logger.info("NLTK check complete.")


# --- Call NLTK download early ---
download_nltk_resources()

# --- Data Classes & Utilities ---


@dataclass
class SentimentResult:
    textblob: str
    vader: str
    bert: str
    textblob_score: float
    vader_score: float
    bert_score: float
    confidence: float


class TextPreprocessor:
    """Handles text cleaning and lemmatization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos(self, treebank_tag: str) -> str:
        tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB,
                   'N': wordnet.NOUN, 'R': wordnet.ADV}
        return tag_map.get(treebank_tag[0], wordnet.NOUN)

    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = ''.join(char for char in text.lower()
                       if char.isalnum() or char.isspace() or char in '.,!?')
        return ' '.join(text.split())

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize, POS tag, lemmatize, remove stopwords."""
        try:
            normalized_text = self.normalize_text(text)
            tokens = word_tokenize(normalized_text)  # This requires 'punkt'
            tagged_tokens = nltk.pos_tag(tokens)
            return [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
                for word, tag in tagged_tokens if word.isalpha() and word not in self.stop_words
            ]
        except LookupError as e:
            logger.error(
                f"NLTK resource missing during tokenization: {e}. Ensure 'punkt' is downloaded.")
            # Return normalized text split by space as a fallback
            return [word for word in self.normalize_text(text).split() if word.isalpha() and word not in self.stop_words]
        except Exception as e:
            logger.error(f"Error during tokenization/lemmatization: {e}")
            return [word for word in self.normalize_text(text).split() if word.isalpha() and word not in self.stop_words]


class DataNormalizer:
    """Handles score normalization."""

    def __init__(self, feature_range: Tuple[int, int] = (1, 5)):
        self.score_scaler = MinMaxScaler(feature_range=feature_range)

    def normalize_scores(self, df: pd.DataFrame, cols_to_normalize: List[str]) -> pd.DataFrame:
        df_normalized = df.copy()
        for col in cols_to_normalize:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if not (df[col].isnull().any() or np.isinf(df[col]).any()):
                    try:
                        shaped_data = df[col].values.reshape(-1, 1)
                        df_normalized[f'{col}_normalized'] = self.score_scaler.fit_transform(
                            shaped_data).flatten()
                    except ValueError as e:
                        logger.error(f"Error normalizing '{col}': {e}")
                        df_normalized[f'{col}_normalized'] = df[col]
                else:
                    logger.warning(
                        f"Skipping normalization for '{col}' due to NaN/Inf values.")
                    df_normalized[f'{col}_normalized'] = df[col]
            else:
                logger.warning(
                    f"Column '{col}' not found or not numeric. Skipping normalization.")
        return df_normalized

# --- Core Analysis Classes ---


class SentimentAnalyzer:
    """Performs sentiment analysis using multiple models."""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = self._load_bert_pipeline()

    @st.cache_resource(show_spinner="Loading BERT sentiment model...")
    def _load_bert_pipeline(_self) -> HFPipeline:
        try:
            logger.info(f"Loading BERT model: {BERT_MODEL_NAME}")
            return pipeline("sentiment-analysis", model=BERT_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}", exc_info=True)
            st.error(
                f"Fatal Error: Could not load BERT model ({BERT_MODEL_NAME}). Error: {e}")
            st.stop()

    def _calculate_confidence(self, scores: List[float]) -> float:
        if not scores or len(scores) < 2:
            return 0.0
        std_dev = np.std(scores)
        return float(1.0 / (1.0 + std_dev))

    def _analyze_textblob(self, text: str) -> Tuple[str, float]:
        try:
            polarity = TextBlob(text).sentiment.polarity
            score = 2.5 * (polarity + 1.0)  # Scale [-1, 1] to [0, 5]
            sentiment = POSITIVE if polarity > 0.1 else NEGATIVE if polarity < -0.1 else NEUTRAL
            return sentiment, score
        except Exception as e:
            # Debug level for less critical errors
            logger.debug(f"TextBlob failed: {e}")
            return NEUTRAL, 2.5

    def _analyze_vader(self, text: str) -> Tuple[str, float]:
        try:
            compound = self.vader_analyzer.polarity_scores(text)['compound']
            score = 2.5 * (compound + 1.0)  # Scale [-1, 1] to [0, 5]
            sentiment = POSITIVE if compound >= 0.05 else NEGATIVE if compound <= -0.05 else NEUTRAL
            return sentiment, score
        except Exception as e:
            logger.debug(f"VADER failed: {e}")
            return NEUTRAL, 2.5

    def _analyze_bert(self, text: str) -> Tuple[str, float]:
        try:
            # Handle potential long text for BERT
            result = self.bert_analyzer(text[:512])[0]
            score = float(result["label"].split()[0])  # Star rating 1-5
            sentiment = POSITIVE if score >= 4 else NEGATIVE if score <= 2 else NEUTRAL
            return sentiment, score
        except Exception as e:
            # Log BERT errors more prominently
            logger.error(f"BERT analysis failed: {e}", exc_info=False)
            return NEUTRAL, 3.0

    def analyze_text(self, text: str) -> SentimentResult:
        if not isinstance(text, str) or not text.strip():
            return SentimentResult(NEUTRAL, NEUTRAL, NEUTRAL, 2.5, 2.5, 3.0, 0.0)

        # Reuse normalized text if needed, but models usually handle raw better
        # normalized_text = self.text_preprocessor.normalize_text(text)
        tb_sentiment, tb_score = self._analyze_textblob(text)
        vader_sentiment, vader_score = self._analyze_vader(text)
        bert_sentiment, bert_score = self._analyze_bert(text)
        confidence = self._calculate_confidence(
            [tb_score, vader_score, bert_score])

        return SentimentResult(
            textblob=tb_sentiment, vader=vader_sentiment, bert=bert_sentiment,
            textblob_score=tb_score, vader_score=vader_score, bert_score=bert_score,
            confidence=confidence
        )


class ReviewAnalyzer:
    """Orchestrates aspect extraction and sentiment analysis for reviews."""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_preprocessor = TextPreprocessor()
        self.data_normalizer = DataNormalizer(feature_range=(1, 5))
        self.aspect_keywords = self._prepare_aspect_keywords(ASPECT_KEYWORDS)
        logger.info("ReviewAnalyzer initialized.")

    def _prepare_aspect_keywords(self, aspect_dict: Dict) -> Dict:
        # (Keep this method as is from previous version)
        processed_aspects = {}
        logger.info("Lemmatizing aspect keywords...")
        for aspect, data in aspect_dict.items():
            processed_aspects[aspect] = {
                "weights": data["weights"], "lemmatized_keywords": {}}
            for category, keywords in data["keywords"].items():
                lemmatized_set = set()
                for keyword in keywords:
                    # Simple lemmatization for keywords
                    lemmatized_words = " ".join(
                        self.text_preprocessor.lemmatizer.lemmatize(kw) for kw in keyword.split())
                    if lemmatized_words:
                        lemmatized_set.add(lemmatized_words)
                processed_aspects[aspect]["lemmatized_keywords"][category] = list(
                    lemmatized_set)
        logger.info("Aspect keyword lemmatization complete.")
        return processed_aspects

    def extract_aspects(self, text: str) -> Dict[str, float]:
        # (Keep this method as is from previous version)
        if not isinstance(text, str) or not text.strip():
            return {}
        lemmatized_tokens = self.text_preprocessor.tokenize_and_lemmatize(text)
        normalized_text_joined = " " + " ".join(lemmatized_tokens) + " "
        detected_aspects = {}
        for aspect, data in self.aspect_keywords.items():
            total_score, hits = 0.0, 0
            for category, lemmatized_keywords in data["lemmatized_keywords"].items():
                weight = data["weights"][category]
                for keyword in lemmatized_keywords:
                    if f" {keyword} " in normalized_text_joined:
                        total_score += weight
                        hits += 1
            if hits > 0:
                detected_aspects[aspect] = total_score / hits
        return detected_aspects

    @st.cache_data(show_spinner="Analyzing review sentiments and aspects...", persist=True)
    def analyze_reviews(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze aspects and sentiments for each review in the DataFrame."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        results = []
        total_rows = len(df)
        progress_bar = st.progress(0, text="Initializing analysis...")
        start_time = time.time()

        # FIX: Use enumerate for progress calculation
        for loop_idx, (df_index, row) in enumerate(df.iterrows()):
            # Calculate progress value safely
            progress_value = (loop_idx + 1) / total_rows
            # Ensure value doesn't exceed 1.0
            progress_value = min(progress_value, 1.0)
            progress_text = f"Processing review {loop_idx + 1}/{total_rows}..."

            if loop_idx % 50 == 0 or loop_idx == total_rows - 1:  # Update even less often
                try:
                    progress_bar.progress(progress_value, text=progress_text)
                except Exception as progress_error:
                    # Log progress bar specific errors but continue analysis
                    logger.warning(
                        f"Progress bar update failed at index {loop_idx}: {progress_error}")

            try:
                review_text = row["text"]
                aspects = _self.extract_aspects(review_text)
                sentiments = _self.sentiment_analyzer.analyze_text(review_text)

                aspect_base = {
                    "hospital": row["title"],
                    "totalScore": row["totalScore"],
                    f"{TEXTBLOB}{SENTIMENT_SUFFIX}": sentiments.textblob,
                    f"{VADER}{SENTIMENT_SUFFIX}": sentiments.vader,
                    f"{BERT}{SENTIMENT_SUFFIX}": sentiments.bert,
                    f"{TEXTBLOB}{RAW_SCORE_SUFFIX}": sentiments.textblob_score,
                    f"{VADER}{RAW_SCORE_SUFFIX}": sentiments.vader_score,
                    f"{BERT}{RAW_SCORE_SUFFIX}": sentiments.bert_score,
                    "sentiment_confidence": sentiments.confidence,
                }

                if not aspects:
                    results.append(
                        {**aspect_base, "aspect": "overall", "aspect_confidence": 0.0})
                else:
                    for aspect, confidence in aspects.items():
                        results.append(
                            {**aspect_base, "aspect": aspect, "aspect_confidence": confidence})

            except Exception as e:
                # Log error with original DataFrame index if helpful for debugging specific rows
                logger.error(
                    f"Error processing review index {df_index}: {e}", exc_info=False)
                continue

        # Final progress update (outside the loop)
        try:
            progress_bar.progress(1.0, text="Analysis complete.")
        except Exception as progress_error:
            logger.warning(
                f"Final progress bar update failed: {progress_error}")

        logger.info(
            f"Analyzed {len(results)} entries in {time.time() - start_time:.2f}s.")
        progress_bar.empty()  # Clear the progress bar

        if not results:
            return pd.DataFrame()
        results_df = pd.DataFrame(results)

        return results_df

# --- AnalysisUtils & Visualizer (Keep as is from previous version) ---


class AnalysisUtils:
    @staticmethod
    def calculate_aspect_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        metrics = []
        grouped = df.groupby(['hospital', 'aspect'])
        bert_sentiment_col = f"{BERT}{SENTIMENT_SUFFIX}"

        for (hospital, aspect), group in grouped:
            metrics.append({
                "hospital": hospital, "aspect": aspect, "mentions": len(group),
                "positive_pct": (group[bert_sentiment_col] == POSITIVE).mean() * 100 if bert_sentiment_col in group else 0,
                "negative_pct": (group[bert_sentiment_col] == NEGATIVE).mean() * 100 if bert_sentiment_col in group else 0,
                "neutral_pct": (group[bert_sentiment_col] == NEUTRAL).mean() * 100 if bert_sentiment_col in group else 0,
                "avg_total_score": group["totalScore"].mean(),
                f"{BERT}_avg_score": group[f"{BERT}{RAW_SCORE_SUFFIX}"].mean() if f"{BERT}{RAW_SCORE_SUFFIX}" in group else np.nan,
                f"{TEXTBLOB}_avg_score": group[f"{TEXTBLOB}{RAW_SCORE_SUFFIX}"].mean() if f"{TEXTBLOB}{RAW_SCORE_SUFFIX}" in group else np.nan,
                f"{VADER}_avg_score": group[f"{VADER}{RAW_SCORE_SUFFIX}"].mean() if f"{VADER}{RAW_SCORE_SUFFIX}" in group else np.nan,
                "avg_aspect_confidence": group["aspect_confidence"].mean(),
                "avg_sentiment_confidence": group["sentiment_confidence"].mean(),
            })
        return pd.DataFrame(metrics).round(2)

    @staticmethod
    def calculate_accuracy_metrics(df: pd.DataFrame) -> Dict:
        metrics = {model: {} for model in SENTIMENT_MODELS}
        if df.empty or 'totalScore' not in df.columns:
            return metrics
        actual_scores = df['totalScore']
        can_correlate = actual_scores.nunique() > 1

        for model in SENTIMENT_MODELS:
            score_col = f'{model}{RAW_SCORE_SUFFIX}'
            if score_col not in df.columns:
                continue
            predicted_scores = df[score_col]
            valid = actual_scores.notna() & predicted_scores.notna()
            if not valid.any():
                continue

            actual_valid, predicted_valid = actual_scores[valid], predicted_scores[valid]
            try:
                mse = mean_squared_error(actual_valid, predicted_valid)
                tolerance_accuracy = np.mean(
                    np.abs(actual_valid - predicted_valid) <= 1.0)
                correlation = actual_valid.corr(
                    predicted_valid) if can_correlate and predicted_valid.nunique() > 1 else None
                metrics[model] = {
                    'mse': mse, 'tolerance_accuracy': tolerance_accuracy,
                    'correlation': correlation if correlation is not None and np.isfinite(correlation) else None
                }
            except Exception as e:
                logger.error(f"Acc Metric Error ({model}): {e}")
        return metrics

    @staticmethod
    def score_to_category(score: float, scale_max: int = 5) -> str:
        if score <= 2.5:
            return NEGATIVE
        elif score <= 3.5:
            return NEUTRAL
        else:
            return POSITIVE

    @staticmethod
    def calculate_confusion_matrices(df: pd.DataFrame) -> Tuple[Dict, List]:
        matrices = {model: np.zeros((3, 3)) for model in SENTIMENT_MODELS}
        categories = SENTIMENT_CATEGORIES
        if df.empty or 'totalScore' not in df.columns:
            return matrices, categories

        try:
            actual_categories = df['totalScore'].apply(
                AnalysisUtils.score_to_category)
        except Exception:
            return matrices, categories

        for model in SENTIMENT_MODELS:
            sentiment_col = f'{model}{SENTIMENT_SUFFIX}'
            if sentiment_col not in df.columns:
                continue
            predicted_categories = df[sentiment_col]
            valid = actual_categories.isin(
                categories) & predicted_categories.isin(categories)
            if not valid.any():
                continue

            try:
                cm = confusion_matrix(
                    actual_categories[valid], predicted_categories[valid], labels=categories)
                sum_rows = cm.sum(axis=1, keepdims=True)
                matrices[model] = cm.astype(
                    'float') / (sum_rows + 1e-6)  # Normalize
            except Exception as e:
                logger.error(f"CM Error ({model}): {e}")
        return matrices, categories


class Visualizer:
    @staticmethod
    @st.cache_data
    def plot_accuracy_metrics(metrics: Dict):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Model Performance Comparison", fontsize=16)
        models = list(metrics.keys())
        plot_data = {
            'MSE (Lower Better)': [metrics[m].get('mse', np.nan) for m in models],
            'Acc (±1pt, Higher Better)': [metrics[m].get('tolerance_accuracy', np.nan) * 100 for m in models],
            'Corr (vs Score, Higher Better)': [metrics[m].get('correlation', np.nan) for m in models]
        }
        ylabels = ['MSE', 'Accuracy (%)', 'Correlation']
        for i, ((title, values), ylabel) in enumerate(zip(plot_data.items(), ylabels)):
            valid_models = [m for m, v in zip(models, values) if pd.notna(v)]
            valid_values = [v for v in values if pd.notna(v)]
            if valid_models:
                sns.barplot(x=valid_models, y=valid_values,
                            ax=axes[i], palette='viridis')
                axes[i].tick_params(axis='x', rotation=15)
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%.2f', fontsize=9)
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[i].set_title(title)
            axes[i].set_ylabel(ylabel)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

    @staticmethod
    @st.cache_data
    def plot_score_comparison(df: pd.DataFrame):
        if df.empty or 'totalScore' not in df.columns:
            return
        num_models = len(SENTIMENT_MODELS)
        fig, axes = plt.subplots(1, num_models, figsize=(
            6 * num_models, 5), sharey=True, sharex=True)
        fig.suptitle("Predicted Score vs. Actual Score", fontsize=16)
        axes = [axes] if num_models == 1 else axes
        actual = df['totalScore']
        min_score, max_score = df['totalScore'].min(
        ), df['totalScore'].max()  # Use actual data range
        # Add buffer, ensure 0-5+ range
        plot_min, plot_max = min(0, min_score - 0.5), max(5.5, max_score + 0.5)

        for idx, model in enumerate(SENTIMENT_MODELS):
            score_col = f'{model}{RAW_SCORE_SUFFIX}'
            if score_col in df.columns:
                predicted = df[score_col]
                valid = actual.notna() & predicted.notna()
                if valid.any():
                    sns.scatterplot(
                        x=actual[valid], y=predicted[valid], ax=axes[idx], alpha=0.4, s=25)
                else:
                    axes[idx].text(0.5, 0.5, 'No valid scores',
                                   ha='center', va='center')
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].plot([plot_min, plot_max], [plot_min, plot_max],
                           'r--', alpha=0.7, label='Ideal Match')
            axes[idx].set_title(f'{model.upper()} vs Actual')
            axes[idx].set_xlabel('Actual Score')
            axes[idx].grid(True, linestyle='--', alpha=0.5)
            axes[idx].legend()
            axes[idx].set_xlim(plot_min, plot_max)
            axes[idx].set_ylim(plot_min, plot_max)
            if idx == 0:
                axes[idx].set_ylabel('Predicted Score')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

    @staticmethod
    @st.cache_data
    def plot_sentiment_distribution(df: pd.DataFrame, title: str, hue_order=SENTIMENT_CATEGORIES):
        if df.empty or 'aspect' not in df.columns:
            return
        num_models = len(SENTIMENT_MODELS)
        fig, axs = plt.subplots(1, num_models, figsize=(
            7 * num_models, 5), sharey=True)
        fig.suptitle(title, fontsize=16)
        axs = [axs] if num_models == 1 else axs
        palettes = ['viridis', 'magma', 'coolwarm']
        for idx, model in enumerate(SENTIMENT_MODELS):
            sentiment_col = f'{model}{SENTIMENT_SUFFIX}'
            if sentiment_col in df.columns:
                plot_data = df.dropna(subset=['aspect', sentiment_col])
                if not plot_data.empty:
                    aspect_order = sorted(plot_data['aspect'].unique())
                    sns.countplot(data=plot_data, y="aspect", hue=sentiment_col, ax=axs[idx], palette=palettes[idx % len(
                        palettes)], order=aspect_order, hue_order=hue_order)
                    axs[idx].legend(title='Sentiment')
                    axs[idx].grid(axis='x', linestyle='--', alpha=0.6)
                else:
                    axs[idx].text(0.5, 0.5, 'No valid data',
                                  ha='center', va='center')
            else:
                axs[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axs[idx].set_title(f'{model.upper()} Sentiment')
            axs[idx].set_ylabel("Aspect" if idx == 0 else "")
            axs[idx].set_xlabel("Count")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

    @staticmethod
    @st.cache_data
    def plot_confusion_matrices(confusion_matrices: Dict, categories: List):
        cols = st.columns(len(SENTIMENT_MODELS))
        for idx, model in enumerate(SENTIMENT_MODELS):
            cm = confusion_matrices.get(model)
            if cm is not None and cm.shape == (len(categories), len(categories)):
                annot_text = [[f'{val:.2f}' for val in row] for row in cm]
                fig = ff.create_annotated_heatmap(z=np.round(
                    cm, 2), x=categories, y=categories, annotation_text=annot_text, colorscale='Viridis', showscale=True)
                fig.update_layout(title=dict(text=f"{model.upper()} CM", x=0.5), xaxis_title="Predicted", yaxis_title="Actual", xaxis={
                                  'side': 'bottom'}, yaxis={'autorange': 'reversed'}, margin=dict(l=50, r=20, t=50, b=50), width=400, height=380)
                with cols[idx]:
                    st.plotly_chart(fig, use_container_width=False)
            else:
                with cols[idx]:
                    st.warning(f"CM data missing/invalid for {model.upper()}.")

# --- Data Loading & Cleaning ---


@st.cache_data(show_spinner="Loading and cleaning data...", persist=True)
def load_data(file):
    # (Keep this function as is from previous version - it includes cleaning)
    try:
        df = pd.read_csv(file)
        logger.info(f"Original DataFrame shape: {df.shape}")
        required_cols = ["text", "totalScore", "title"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"CSV missing required columns: {', '.join(missing)}.")
            return None

        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        logger.info(
            f"Dropped {initial_rows - len(df)} rows with missing required values.")

        df['text'] = df['text'].astype(str).str.strip()
        df['title'] = df['title'].astype(str).str.strip()
        df['totalScore'] = pd.to_numeric(df['totalScore'], errors='coerce')

        initial_rows = len(df)
        df.dropna(subset=['totalScore'], inplace=True)
        logger.info(
            f"Dropped {initial_rows - len(df)} rows with non-numeric 'totalScore'.")

        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} exact duplicate rows.")

        initial_rows = len(df)
        df.drop_duplicates(subset=['title', 'text'],
                           keep='first', inplace=True)
        logger.info(
            f"Dropped {initial_rows - len(df)} duplicate reviews (same text/title).")

        initial_rows = len(df)
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df = df[df['word_count'] >= MIN_REVIEW_WORDS]
        df.drop(columns=['word_count'], inplace=True)
        logger.info(
            f"Dropped {initial_rows - len(df)} rows with < {MIN_REVIEW_WORDS} words.")

        logger.info(f"Cleaned DataFrame shape: {df.shape}")
        if df.empty:
            st.error("No valid data remaining after cleaning.")
            return None
        return df

    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV is empty.")
        return None
    except Exception as e:
        st.error(f"Error reading/cleaning CSV: {e}")
        logger.error(f"CSV Error: {e}", exc_info=True)
        return None


# --- Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Hospital Review Analyzer")
    st.title("🏥 Hospital Review Analysis")
    st.markdown("Upload a CSV with `text`, `totalScore`, and `title` columns.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is None:
        st.info("Please upload a CSV file.")
        return

    input_df = load_data(uploaded_file)
    if input_df is None or input_df.empty:
        return

    try:
        analyzer = ReviewAnalyzer()
        analyzed_df = analyzer.analyze_reviews(input_df)

        if analyzed_df.empty:
            st.error("Analysis produced no results. Check input data quality.")
            return

        # --- Metrics Calculation ---
        with st.spinner("Calculating metrics..."):
            aspect_metrics_df = AnalysisUtils.calculate_aspect_metrics(
                analyzed_df)
            accuracy_metrics = AnalysisUtils.calculate_accuracy_metrics(
                analyzed_df)
            confusion_matrices, categories = AnalysisUtils.calculate_confusion_matrices(
                analyzed_df)

        st.success("Analysis Complete!")
        st.markdown("---")

        # --- Results Display ---
        st.header("📊 Model Performance Evaluation")
        tab_perf1, tab_perf2, tab_perf3 = st.tabs(
            ["Confusion Matrices", "Accuracy Metrics", "Score Comparison"])

        with tab_perf1:
            st.subheader("Confusion Matrix Analysis")
            st.markdown(
                "Compares model predictions vs. actual score categories (Normalized by Row). Diagonal = agreement.")
            Visualizer.plot_confusion_matrices(confusion_matrices, categories)

        with tab_perf2:
            st.subheader("Accuracy Metrics")
            metrics_display = pd.DataFrame({
                model: {
                    'MSE': f"{mets.get('mse', np.nan):.3f}",
                    'Acc (±1pt)': f"{mets.get('tolerance_accuracy', 0)*100:.1f}%",
                    'Corr': f"{mets.get('correlation', np.nan):.3f}"
                } for model, mets in accuracy_metrics.items() if mets
            }).transpose().fillna("N/A")
            metrics_display.index.name = "Model"
            st.table(metrics_display)
            Visualizer.plot_accuracy_metrics(accuracy_metrics)

        with tab_perf3:
            st.subheader("Predicted vs Actual Score Scatter Plot")
            Visualizer.plot_score_comparison(analyzed_df)

        st.markdown("---")
        st.header("🔬 Sentiment & Aspect Analysis")

        tab_overall, tab_hospital = st.tabs(
            ["Overall Analysis", "Hospital Specific"])

        with tab_overall:
            st.subheader("Overall Sentiment Distribution by Aspect")
            Visualizer.plot_sentiment_distribution(
                analyzed_df, "Overall Sentiment Distribution")

            st.subheader("Detailed Aspect Metrics (All Hospitals)")
            if not aspect_metrics_df.empty:
                sort_options_overall = {"mentions": "# Mentions", "positive_pct": "+ve %",
                                        "negative_pct": "-ve %", "avg_total_score": "Avg Score"}
                valid_sort_keys_overall = [
                    k for k in sort_options_overall.keys() if k in aspect_metrics_df.columns]
                if valid_sort_keys_overall:
                    sort_metric_overall = st.selectbox(
                        "Sort by:", valid_sort_keys_overall, format_func=lambda x: sort_options_overall.get(x, x), key="sort_overall")
                    display_metrics_overall = aspect_metrics_df.sort_values(
                        by=sort_metric_overall, ascending=False)
                else:
                    display_metrics_overall = aspect_metrics_df  # No valid sort key

                # Use st.dataframe for better display control
                st.dataframe(display_metrics_overall.style.format({
                    col: "{:.1f}%" for col in ['positive_pct', 'negative_pct', 'neutral_pct'] if col in display_metrics_overall}
                    # Handle NaN formatting
                    | {col: "{:.2f}" for col in display_metrics_overall.select_dtypes(include=np.number).columns if 'pct' not in col}, na_rep='N/A'),
                    use_container_width=True)

            else:
                st.warning("No aspect metrics calculated.")

        with tab_hospital:
            available_hospitals = sorted(input_df["title"].unique())
            if not available_hospitals:
                st.warning("No unique hospital names found.")
            else:
                selected_hospital = st.selectbox(
                    "Select Hospital:", available_hospitals, key="hospital_select")
                if selected_hospital:
                    hospital_analyzed_data = analyzed_df[analyzed_df["hospital"]
                                                         == selected_hospital]
                    hospital_metrics_data = aspect_metrics_df[aspect_metrics_df["hospital"]
                                                              == selected_hospital]

                    if hospital_analyzed_data.empty:
                        st.info(
                            f"No analyzed data found for {selected_hospital}.")
                    else:
                        st.subheader(f"Analysis for: {selected_hospital}")
                        Visualizer.plot_sentiment_distribution(
                            hospital_analyzed_data, f"Sentiment Distribution for {selected_hospital}")

                        st.subheader("Aspect Metrics")
                        if not hospital_metrics_data.empty:
                            st.dataframe(hospital_metrics_data.style.format({
                                col: "{:.1f}%" for col in ['positive_pct', 'negative_pct', 'neutral_pct'] if col in hospital_metrics_data}
                                | {col: "{:.2f}" for col in hospital_metrics_data.select_dtypes(include=np.number).columns if 'pct' not in col}, na_rep='N/A'),
                                use_container_width=True)
                        else:
                            st.info(
                                "No specific aspect metrics for this hospital.")

    except Exception as e:
        # Display the primary error clearly if it's StreamlitAPIException
        if isinstance(e, st.errors.StreamlitAPIException) and "Progress Value has invalid value" in str(e):
            st.error(
                f"Analysis stopped due to a progress bar calculation issue: {e}")
            logger.error(
                f"Progress bar error prevented full analysis: {e}", exc_info=False)
        else:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            logger.error(
                "Unhandled exception in main app flow:", exc_info=True)
        st.info("If errors persist, please check the console logs for more details.")


if __name__ == "__main__":
    main()
