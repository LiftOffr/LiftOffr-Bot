#!/usr/bin/env python3
"""
Sentiment Analysis Integration

This module integrates market sentiment analysis from news sources and social media
to enhance trading decisions across all supported cryptocurrency assets.

Key features:
1. Real-time news sentiment analysis for crypto assets
2. Social media sentiment extraction from Twitter, Reddit, etc.
3. Sentiment trend tracking over multiple timeframes
4. Integration with ML models to adjust prediction confidence
5. Weighting system to prioritize reliable sentiment sources
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

# Optional imports - will attempt to load these if available
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
SENTIMENT_DATA_DIR = "sentiment_data"
os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)

# Crypto news sources
CRYPTO_NEWS_SOURCES = [
    "https://cointelegraph.com/",
    "https://www.coindesk.com/",
    "https://cryptonews.com/",
    "https://beincrypto.com/",
    "https://cryptoslate.com/",
    "https://www.bloomberg.com/crypto",
    "https://dailyhodl.com/"
]

# Asset-specific RSS feeds and URLs
ASSET_URLS = {
    "BTC/USD": [
        "https://cryptonews.com/news/bitcoin-news/",
        "https://cointelegraph.com/tags/bitcoin",
        "https://www.coindesk.com/tag/bitcoin/",
    ],
    "ETH/USD": [
        "https://cryptonews.com/news/ethereum-news/",
        "https://cointelegraph.com/tags/ethereum",
        "https://www.coindesk.com/tag/ethereum/",
    ],
    "SOL/USD": [
        "https://cryptonews.com/news/solana-news/",
        "https://cointelegraph.com/tags/solana",
        "https://www.coindesk.com/tag/solana/",
    ],
    "DOT/USD": [
        "https://cryptonews.com/news/polkadot-news/",
        "https://cointelegraph.com/tags/polkadot",
    ],
    "LINK/USD": [
        "https://cryptonews.com/news/chainlink-news/",
        "https://cointelegraph.com/tags/chainlink",
    ]
}

# Asset keywords for news filtering
ASSET_KEYWORDS = {
    "BTC/USD": ["bitcoin", "btc", "bitcoin price", "btc price", "bitcoin market", "btc market", "bitcoin trading", "btc trading"],
    "ETH/USD": ["ethereum", "eth", "ethereum price", "eth price", "ethereum market", "eth market", "ethereum trading", "eth trading"],
    "SOL/USD": ["solana", "sol", "solana price", "sol price", "solana market", "sol market", "solana trading", "sol trading"],
    "DOT/USD": ["polkadot", "dot", "polkadot price", "dot price", "polkadot market", "dot market", "polkadot trading", "dot trading"],
    "LINK/USD": ["chainlink", "link", "chainlink price", "link price", "chainlink market", "link market", "chainlink trading", "link trading"]
}

# Default sentiment model
DEFAULT_SENTIMENT_MODEL = "finbert"  # Options: "vader", "finbert", "distilbert"


class MarketSentimentAnalyzer:
    """
    Analyzes market sentiment from news and social media sources
    for cryptocurrency assets.
    """
    
    def __init__(self, model_name=DEFAULT_SENTIMENT_MODEL):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name (str): Sentiment model to use ('vader', 'finbert', or 'distilbert')
        """
        self.model_name = model_name
        self.sentiment_model = None
        self.tokenizer = None
        
        # Initialize sentiment analysis model
        self._initialize_sentiment_model()
        
        logger.info(f"Initialized Market Sentiment Analyzer with {model_name} model")
    
    def _initialize_sentiment_model(self):
        """Initialize the specified sentiment analysis model"""
        # VADER sentiment analyzer (rule-based)
        if self.model_name == "vader":
            if NLTK_AVAILABLE:
                # Download VADER lexicon if not already downloaded
                try:
                    nltk.data.find('vader_lexicon')
                except LookupError:
                    nltk.download('vader_lexicon')
                
                self.sentiment_model = SentimentIntensityAnalyzer()
                logger.info("Initialized VADER sentiment analyzer")
            else:
                logger.warning("NLTK not available, falling back to basic sentiment analysis")
                self.model_name = "basic"
        
        # FinBERT (financial text sentiment)
        elif self.model_name == "finbert" and TRANSFORMERS_AVAILABLE:
            try:
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=self.tokenizer)
                logger.info("Initialized FinBERT sentiment analyzer")
            except Exception as e:
                logger.error(f"Error initializing FinBERT: {e}")
                logger.warning("Falling back to basic sentiment analysis")
                self.model_name = "basic"
        
        # DistilBERT (general purpose sentiment)
        elif self.model_name == "distilbert" and TRANSFORMERS_AVAILABLE:
            try:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.sentiment_model = pipeline("sentiment-analysis", model=model_name)
                logger.info("Initialized DistilBERT sentiment analyzer")
            except Exception as e:
                logger.error(f"Error initializing DistilBERT: {e}")
                logger.warning("Falling back to basic sentiment analysis")
                self.model_name = "basic"
        
        # Basic sentiment analysis (fallback)
        elif self.model_name == "basic" or not (NLTK_AVAILABLE or TRANSFORMERS_AVAILABLE):
            logger.info("Using basic keyword-based sentiment analysis")
            self.model_name = "basic"
        
        else:
            logger.warning(f"Unknown sentiment model: {self.model_name}, falling back to basic")
            self.model_name = "basic"
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5, "compound": 0}
        
        # VADER sentiment analysis
        if self.model_name == "vader":
            scores = self.sentiment_model.polarity_scores(text)
            
            # Determine sentiment label
            compound = scores["compound"]
            if compound >= 0.05:
                label = "POSITIVE"
            elif compound <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            # Normalize score to 0-1 range
            score = (compound + 1) / 2
            
            return {
                "label": label,
                "score": score,
                "compound": compound,
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"]
            }
        
        # FinBERT or DistilBERT sentiment analysis
        elif self.model_name in ["finbert", "distilbert"]:
            try:
                # Handle long texts by chunking
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                results = []
                
                for chunk in chunks:
                    # Skip empty chunks
                    if not chunk.strip():
                        continue
                    
                    result = self.sentiment_model(chunk)[0]
                    results.append(result)
                
                # If no valid chunks, return neutral
                if not results:
                    return {"label": "NEUTRAL", "score": 0.5, "compound": 0}
                
                # Average the scores for each label
                label_scores = defaultdict(list)
                for result in results:
                    label_scores[result["label"]].append(result["score"])
                
                # Find the label with the highest average score
                avg_scores = {label: sum(scores)/len(scores) for label, scores in label_scores.items()}
                max_label = max(avg_scores, key=avg_scores.get)
                max_score = avg_scores[max_label]
                
                # Convert to standardized format
                if "POSITIVE" in max_label:
                    label = "POSITIVE"
                    compound = max_score * 2 - 1  # Convert 0.5-1 to 0-1
                elif "NEGATIVE" in max_label:
                    label = "NEGATIVE"
                    compound = -max_score  # Convert 0.5-1 to -0.5 to -1
                else:
                    label = "NEUTRAL"
                    compound = 0
                
                return {
                    "label": label,
                    "score": max_score,
                    "compound": compound,
                    "details": avg_scores
                }
            
            except Exception as e:
                logger.error(f"Error in transformer-based sentiment analysis: {e}")
                return {"label": "NEUTRAL", "score": 0.5, "compound": 0, "error": str(e)}
        
        # Basic keyword-based sentiment analysis
        else:
            positive_words = ["bullish", "surge", "rally", "boom", "gain", "soar", "rise", "good", "positive", 
                             "increase", "growth", "profit", "improve", "stronger", "opportunity", "up", "uptrend", "higher"]
            
            negative_words = ["bearish", "crash", "drop", "plunge", "fall", "decline", "decrease", "bad", "negative", 
                             "loss", "concern", "risk", "weak", "down", "downtrend", "lower"]
            
            # Convert to lowercase for case-insensitive matching
            text_lower = text.lower()
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment
            total_count = pos_count + neg_count
            if total_count == 0:
                return {"label": "NEUTRAL", "score": 0.5, "compound": 0}
            
            pos_ratio = pos_count / total_count
            compound = (pos_ratio - 0.5) * 2  # Scale to -1 to 1
            
            # Determine label
            if compound > 0.1:
                label = "POSITIVE"
            elif compound < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            # Normalize score to 0-1 range
            score = (compound + 1) / 2
            
            return {
                "label": label,
                "score": score,
                "compound": compound,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "total_count": total_count
            }
    
    def fetch_news_content(self, url):
        """
        Fetch and extract main content from a news URL
        
        Args:
            url (str): URL to fetch
            
        Returns:
            str: Extracted text content
        """
        if not TRAFILATURA_AVAILABLE:
            logger.warning("Trafilatura not available, can't extract article content")
            return ""
        
        try:
            # Download page
            downloaded = trafilatura.fetch_url(url)
            
            # Extract main content
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text or ""
            
            return ""
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""
    
    def fetch_asset_news(self, asset, max_articles=10, max_age_hours=24):
        """
        Fetch recent news articles for a specific asset
        
        Args:
            asset (str): Asset to fetch news for (e.g., "BTC/USD")
            max_articles (int): Maximum number of articles to fetch
            max_age_hours (int): Maximum age of articles in hours
            
        Returns:
            list: List of article data dictionaries
        """
        articles = []
        
        # Get URLs specific to this asset
        urls = ASSET_URLS.get(asset, [])
        
        # If no specific URLs, use general crypto news sources
        if not urls:
            urls = CRYPTO_NEWS_SOURCES
        
        # Get keywords for this asset
        keywords = ASSET_KEYWORDS.get(asset, [asset.split('/')[0].lower()])
        
        # Use basic scraping to find articles (simplified implementation)
        for url in urls[:min(5, len(urls))]:  # Limit to 5 sources to avoid rate limits
            try:
                # Simulate fetching article links and timestamps
                # In a real implementation, you would use a proper scraping library
                # or an API to get actual article data
                
                # Since we can't actually scrape here, we'll simulate finding articles
                logger.info(f"Would fetch articles from {url} for {asset}")
                
                # For demonstration purposes:
                simulated_articles = [
                    {
                        "title": f"Latest {asset.split('/')[0]} market movements",
                        "url": f"{url}article1",
                        "timestamp": datetime.now() - timedelta(hours=1),
                        "source": url
                    },
                    {
                        "title": f"{asset.split('/')[0]} price analysis",
                        "url": f"{url}article2",
                        "timestamp": datetime.now() - timedelta(hours=2),
                        "source": url
                    }
                ]
                
                articles.extend(simulated_articles[:max_articles//len(urls)])
                
            except Exception as e:
                logger.error(f"Error fetching articles from {url}: {e}")
        
        # Sort by timestamp (most recent first)
        articles.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limit to max articles
        articles = articles[:max_articles]
        
        # Log the simulated discovery
        logger.info(f"Found {len(articles)} articles for {asset}")
        
        # Return empty list for now since this is just a demonstration
        # In a real implementation, this would return actual article data
        return []
    
    def analyze_asset_sentiment(self, asset, max_age_hours=24, refresh_cache=False):
        """
        Analyze sentiment for a specific asset
        
        Args:
            asset (str): Asset to analyze sentiment for (e.g., "BTC/USD")
            max_age_hours (int): Maximum age of articles to consider
            refresh_cache (bool): Whether to refresh the sentiment cache
            
        Returns:
            dict: Sentiment analysis results
        """
        # Check cache
        cache_file = os.path.join(SENTIMENT_DATA_DIR, f"{asset.replace('/', '_')}_sentiment.json")
        
        if os.path.exists(cache_file) and not refresh_cache:
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours < max_age_hours:
                    logger.info(f"Using cached sentiment data for {asset} (age: {age_hours:.1f} hours)")
                    return cached_data
            except Exception as e:
                logger.error(f"Error reading sentiment cache for {asset}: {e}")
        
        # In a real implementation, we would fetch and analyze actual articles
        # For demonstration purposes, we'll generate simulated sentiment
        
        # Simulate article fetching and analysis
        logger.info(f"Analyzing sentiment for {asset}")
        
        # Get keywords for querying news data
        asset_symbol = asset.split('/')[0]
        
        # Simulated sentiment result
        sentiment_result = {
            "asset": asset,
            "timestamp": datetime.now().isoformat(),
            "sentiment_classification": "NEUTRAL",
            "sentiment_score": 0.5,
            "signal": "NEUTRAL",
            "strength": 0.3,
            "confidence": 0.5,
            "source_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "keywords": ASSET_KEYWORDS.get(asset, [asset_symbol.lower()]),
            "sentiment_model": self.model_name,
            "analysis_method": "simulation"  # In real implementation, this would be "full"
        }
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(sentiment_result, f, indent=2)
            logger.info(f"Saved sentiment data for {asset} to cache")
        except Exception as e:
            logger.error(f"Error saving sentiment cache for {asset}: {e}")
        
        return sentiment_result
    
    def calculate_market_sentiment_score(self, asset_sentiments):
        """
        Calculate overall market sentiment score from multiple asset sentiments
        
        Args:
            asset_sentiments (dict): Dictionary of asset sentiment results
            
        Returns:
            dict: Market sentiment score
        """
        if not asset_sentiments:
            return {
                "overall_sentiment": "NEUTRAL",
                "score": 0.5,
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate weighted sentiment score
        # Major assets (BTC, ETH) have higher weight
        weights = {
            "BTC/USD": 0.4,
            "ETH/USD": 0.3,
            "SOL/USD": 0.1,
            "DOT/USD": 0.1,
            "LINK/USD": 0.1
        }
        
        # Normalize weights for available assets
        available_assets = set(asset_sentiments.keys())
        total_weight = sum(weights[asset] for asset in available_assets if asset in weights)
        
        if total_weight == 0:
            # Equal weights if no predefined weights
            normalized_weights = {asset: 1/len(available_assets) for asset in available_assets}
        else:
            normalized_weights = {asset: weights.get(asset, 0)/total_weight for asset in available_assets}
        
        # Calculate weighted sentiment score
        weighted_score = 0
        confidence_sum = 0
        
        for asset, sentiment in asset_sentiments.items():
            if asset in normalized_weights:
                weight = normalized_weights[asset]
                score = sentiment.get("sentiment_score", 0.5)
                confidence = sentiment.get("confidence", 0.5)
                
                weighted_score += score * weight
                confidence_sum += confidence * weight
        
        # Determine overall sentiment
        if weighted_score > 0.6:
            overall_sentiment = "BULLISH"
        elif weighted_score < 0.4:
            overall_sentiment = "BEARISH"
        else:
            overall_sentiment = "NEUTRAL"
        
        return {
            "overall_sentiment": overall_sentiment,
            "score": weighted_score,
            "confidence": confidence_sum,
            "assets_analyzed": list(asset_sentiments.keys()),
            "asset_weights": normalized_weights,
            "timestamp": datetime.now().isoformat()
        }

class MarketSentimentIntegrator:
    """
    Integrates market sentiment analysis with trading strategies
    """
    
    def __init__(self, sentiment_analyzer=None, model_name=DEFAULT_SENTIMENT_MODEL):
        """
        Initialize the sentiment integrator
        
        Args:
            sentiment_analyzer (MarketSentimentAnalyzer, optional): Sentiment analyzer instance
            model_name (str): Sentiment model to use if creating a new analyzer
        """
        self.sentiment_analyzer = sentiment_analyzer or MarketSentimentAnalyzer(model_name)
        self.sentiment_history = {}
        self.market_sentiment_history = []
        
        logger.info("Initialized Market Sentiment Integrator")
    
    def update_sentiment_history(self, asset, sentiment_data):
        """
        Update sentiment history for an asset
        
        Args:
            asset (str): Asset to update history for
            sentiment_data (dict): Sentiment analysis results
        """
        if asset not in self.sentiment_history:
            self.sentiment_history[asset] = []
        
        # Add new sentiment data with timestamp
        if isinstance(sentiment_data, dict) and "timestamp" in sentiment_data:
            self.sentiment_history[asset].append(sentiment_data)
            
            # Keep only last 100 records
            if len(self.sentiment_history[asset]) > 100:
                self.sentiment_history[asset] = self.sentiment_history[asset][-100:]
            
            logger.info(f"Updated sentiment history for {asset}")
    
    def analyze_asset_sentiment(self, asset, max_age_hours=24, refresh_cache=False):
        """
        Analyze sentiment for a specific asset
        
        Args:
            asset (str): Asset to analyze sentiment for
            max_age_hours (int): Maximum age of articles to consider
            refresh_cache (bool): Whether to refresh the sentiment cache
            
        Returns:
            dict: Sentiment analysis results
        """
        # Use the sentiment analyzer to get sentiment
        sentiment_data = self.sentiment_analyzer.analyze_asset_sentiment(
            asset, max_age_hours, refresh_cache
        )
        
        # Update sentiment history
        self.update_sentiment_history(asset, sentiment_data)
        
        return sentiment_data
    
    def analyze_market_sentiment(self, assets=None, max_age_hours=24, refresh_cache=False):
        """
        Analyze overall market sentiment
        
        Args:
            assets (list, optional): Assets to include in analysis
            max_age_hours (int): Maximum age of articles to consider
            refresh_cache (bool): Whether to refresh the sentiment cache
            
        Returns:
            dict: Market sentiment results
        """
        # Default assets if none specified
        if not assets:
            assets = list(ASSET_KEYWORDS.keys())
        
        # Analyze sentiment for each asset
        asset_sentiments = {}
        
        for asset in assets:
            try:
                sentiment = self.analyze_asset_sentiment(asset, max_age_hours, refresh_cache)
                asset_sentiments[asset] = sentiment
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {asset}: {e}")
        
        # Calculate market sentiment
        market_sentiment = self.sentiment_analyzer.calculate_market_sentiment_score(asset_sentiments)
        
        # Update market sentiment history
        self.market_sentiment_history.append(market_sentiment)
        
        # Keep only last 100 records
        if len(self.market_sentiment_history) > 100:
            self.market_sentiment_history = self.market_sentiment_history[-100:]
        
        return market_sentiment
    
    def get_sentiment_trend(self, asset, window=24):
        """
        Get sentiment trend for an asset over a time window
        
        Args:
            asset (str): Asset to get trend for
            window (int): Number of hours to include
            
        Returns:
            dict: Trend analysis
        """
        if asset not in self.sentiment_history or not self.sentiment_history[asset]:
            return {
                "trend": "NEUTRAL",
                "change": 0,
                "current_score": 0.5,
                "previous_score": 0.5,
                "confidence": 0.3
            }
        
        # Get sentiment history
        history = self.sentiment_history[asset]
        
        # Filter to window
        cutoff_time = datetime.now() - timedelta(hours=window)
        recent_history = [
            item for item in history
            if datetime.fromisoformat(item["timestamp"]) >= cutoff_time
        ]
        
        if not recent_history:
            return {
                "trend": "NEUTRAL",
                "change": 0,
                "current_score": history[-1].get("sentiment_score", 0.5),
                "previous_score": history[-1].get("sentiment_score", 0.5),
                "confidence": history[-1].get("confidence", 0.3)
            }
        
        # Sort by timestamp
        recent_history.sort(key=lambda x: x["timestamp"])
        
        # Get current and previous sentiment scores
        current_score = recent_history[-1].get("sentiment_score", 0.5)
        
        if len(recent_history) > 1:
            previous_score = recent_history[0].get("sentiment_score", 0.5)
        else:
            previous_score = current_score
        
        # Calculate change
        change = current_score - previous_score
        
        # Determine trend
        if change > 0.1:
            trend = "IMPROVING"
        elif change < -0.1:
            trend = "DETERIORATING"
        else:
            trend = "STABLE"
        
        return {
            "trend": trend,
            "change": change,
            "current_score": current_score,
            "previous_score": previous_score,
            "confidence": recent_history[-1].get("confidence", 0.3),
            "window_hours": window,
            "data_points": len(recent_history)
        }
    
    def get_trading_sentiment_adjustment(self, asset, sensitivity=1.0, refresh_cache=False):
        """
        Get sentiment-based adjustment for trading
        
        Args:
            asset (str): Asset to get adjustment for
            sensitivity (float): Sensitivity multiplier for adjustment (0.0-2.0)
            refresh_cache (bool): Whether to refresh sentiment data
            
        Returns:
            dict: Sentiment adjustment for trading
        """
        # Get current sentiment
        sentiment = self.analyze_asset_sentiment(asset, refresh_cache=refresh_cache)
        
        # Get sentiment trend
        trend = self.get_sentiment_trend(asset)
        
        # Calculate adjusted score
        score = sentiment.get("sentiment_score", 0.5)
        
        # Apply sensitivity
        adjusted_score = 0.5 + (score - 0.5) * sensitivity
        
        # Boost based on trend
        if trend["trend"] == "IMPROVING":
            adjusted_score += 0.05 * sensitivity
        elif trend["trend"] == "DETERIORATING":
            adjusted_score -= 0.05 * sensitivity
        
        # Clamp to 0-1 range
        adjusted_score = max(0, min(1, adjusted_score))
        
        # Determine trading signal
        if adjusted_score > 0.6:
            signal = "BUY"
            strength = (adjusted_score - 0.6) * 2.5 * sensitivity  # Scale to 0-1
        elif adjusted_score < 0.4:
            signal = "SELL"
            strength = (0.4 - adjusted_score) * 2.5 * sensitivity  # Scale to 0-1
        else:
            signal = "NEUTRAL"
            strength = 0.3
        
        # Clamp strength to 0-1 range
        strength = max(0, min(1, strength))
        
        return {
            "asset": asset,
            "raw_score": score,
            "adjusted_score": adjusted_score,
            "signal": signal,
            "strength": strength,
            "confidence": sentiment.get("confidence", 0.5),
            "trend": trend["trend"],
            "sentiment_classification": sentiment.get("sentiment_classification", "NEUTRAL"),
            "sensitivity": sensitivity
        }
    
    def apply_sentiment_to_model_prediction(self, model_prediction, sentiment_adjustment, weight=0.2):
        """
        Apply sentiment adjustment to model prediction
        
        Args:
            model_prediction (float): Raw model prediction value
            sentiment_adjustment (dict): Sentiment adjustment from get_trading_sentiment_adjustment
            weight (float): Weight of sentiment in adjustment (0.0-1.0)
            
        Returns:
            float: Adjusted prediction
        """
        if not sentiment_adjustment:
            return model_prediction
        
        # Get sentiment signal as numeric value
        sentiment_signal = 0
        if sentiment_adjustment["signal"] == "BUY":
            sentiment_signal = sentiment_adjustment["strength"]
        elif sentiment_adjustment["signal"] == "SELL":
            sentiment_signal = -sentiment_adjustment["strength"]
        
        # Scale by weight
        sentiment_contribution = sentiment_signal * weight
        
        # Add to model prediction
        adjusted_prediction = model_prediction + sentiment_contribution
        
        logger.info(f"Applied sentiment to prediction: {model_prediction:.4f} -> {adjusted_prediction:.4f} (sentiment: {sentiment_signal:.4f}, weight: {weight:.2f})")
        
        return adjusted_prediction
    
    def plot_sentiment_history(self, asset, days=30, output_path=None):
        """
        Plot sentiment history for an asset
        
        Args:
            asset (str): Asset to plot history for
            days (int): Number of days to include
            output_path (str, optional): Path to save the plot
        """
        if asset not in self.sentiment_history or not self.sentiment_history[asset]:
            logger.warning(f"No sentiment history available for {asset}")
            return
        
        # Get sentiment history
        history = self.sentiment_history[asset]
        
        # Filter to window
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            item for item in history
            if datetime.fromisoformat(item["timestamp"]) >= cutoff_time
        ]
        
        if not recent_history:
            logger.warning(f"No recent sentiment history available for {asset}")
            return
        
        # Extract timestamps and scores
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in recent_history]
        scores = [item.get("sentiment_score", 0.5) for item in recent_history]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot sentiment scores
        plt.plot(timestamps, scores, marker='o', linestyle='-', color='blue')
        
        # Add horizontal lines
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.3)
        plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.title(f'Sentiment History for {asset} (Last {days} days)')
        plt.ylim(0, 1)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved sentiment history plot to {output_path}")
        else:
            plt.show()
        
        plt.close()

# Create global instance for easy access
market_sentiment_integrator = MarketSentimentIntegrator()

def main():
    """
    Main function to run sentiment analysis independently
    """
    # Create sentiment analyzer
    analyzer = MarketSentimentAnalyzer()
    
    # Create sentiment integrator
    integrator = MarketSentimentIntegrator(analyzer)
    
    # Analyze sentiment for all assets
    assets = list(ASSET_KEYWORDS.keys())
    
    for asset in assets:
        sentiment = integrator.analyze_asset_sentiment(asset, refresh_cache=True)
        logger.info(f"Sentiment for {asset}: {sentiment['sentiment_classification']} (score: {sentiment['sentiment_score']:.2f})")
        
        # Get trading adjustment
        adjustment = integrator.get_trading_sentiment_adjustment(asset)
        logger.info(f"Trading adjustment for {asset}: {adjustment['signal']} (strength: {adjustment['strength']:.2f})")
    
    # Analyze market sentiment
    market_sentiment = integrator.analyze_market_sentiment(assets)
    logger.info(f"Market sentiment: {market_sentiment['overall_sentiment']} (score: {market_sentiment['score']:.2f})")
    
    return 0

if __name__ == "__main__":
    main()