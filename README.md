[![Sentiment Analysis](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=00E5FF&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=Sentiment+Analysis;NLP+%2B+Machine+Learning;End-to-End+ML+Project+2026)](https://git.io/typing-svg)  
[![Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![NLTK-3.8%2B-013243?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/NLTK-3.8%2B-013243?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/) [![TextBlob-0.17%2B-FF6F00?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/TextBlob-0.17%2B-FF6F00?style=for-the-badge&logo=python&logoColor=white)](https://textblob.readthedocs.io/) [![WordCloud-1.9%2B-Fcdc00?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/WordCloud-1.9%2B-Fcdc00?style=for-the-badge&logo=python&logoColor=white)](https://amueller.github.io/word_cloud/)  
[![GitHub stars](https://img.shields.io/github/stars/basitali08/sentiment-analysis?style=social)](https://github.com/basitali08/sentiment-analysis) [![GitHub forks](https://img.shields.io/github/forks/basitali08/sentiment-analysis?style=social)](https://github.com/basitali08/sentiment-analysis)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## 🔍 Overview

> **Classify product reviews as positive, negative, or neutral with 86% accuracy using NLP & Machine Learning.**

Manual review analysis is impossible at scale. This project builds a complete NLP system that:

- Analyzes **3,000+ product reviews** for sentiment patterns
- Uses **TF-IDF** for intelligent feature extraction
- Compares **Naive Bayes, Logistic Regression & SVM**
- Achieves **86% accuracy** with the best model
- Provides **real-time sentiment prediction** for new reviews

[![Accuracy](https://img.shields.io/badge/Accuracy-86%25-success?style=flat-square&labelColor=1a1a2e&color=00e676)](https://github.com/basitali08/sentiment-analysis)
[![Precision](https://img.shields.io/badge/Precision-85%25-success?style=flat-square&labelColor=1a1a2e&color=00e676)](https://github.com/basitali08/sentiment-analysis)
[![Recall](https://img.shields.io/badge/Recall-84%25-success?style=flat-square&labelColor=1a1a2e&color=00e676)](https://github.com/basitali08/sentiment-analysis)
[![F1-Score](https://img.shields.io/badge/F1--Score-84%25-success?style=flat-square&labelColor=1a1a2e&color=00e676)](https://github.com/basitali08/sentiment-analysis)
[![Dataset](https://img.shields.io/badge/Dataset-3%2C000_reviews-blue?style=flat-square&labelColor=1a1a2e&color=0066ff)](https://github.com/basitali08/sentiment-analysis)

---

## 📊 Dataset

**Source:** Product Reviews Dataset (Kaggle)

| Feature | Type | Description |
|---------|------|-------------|
| `review_id` | Numeric | Unique review identifier |
| `review_text` | Text | Customer review content |
| `rating` | Numeric | 1-5 star rating |
| `sentiment` | **Target** | **Positive**, **Negative**, **Neutral** |
| `product_category` | Categorical | Electronics, Clothing, Home, Books |

> **Distribution:** 40% Positive, 30% Negative, 30% Neutral — balanced dataset.

---

## 🔄 Project Pipeline

```
📥 Load Data → 🔍 EDA Viz → 🧹 Clean Text → 📝 Tokenize & Lemmatize → 📊 TF-IDF → 🤖 Train 3 Models → 📊 Evaluate → 💾 Save Model
```

**Step-by-step:**

1. **Data Loading** — Load CSV with 3000+ reviews
2. **EDA** — Visualize sentiment distribution, word clouds, review lengths
3. **Text Preprocessing** — Lowercase, remove special chars, tokenize, lemmatize
4. **Feature Extraction** — Convert text to TF-IDF vectors
5. **Model Training** — Train Naive Bayes, Logistic Regression, SVM
6. **Evaluation** — Compare accuracy, precision, recall, F1
7. **Prediction Function** — Build real-time sentiment predictor

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 78% | 76% | 75% | 75% |
| Logistic Regression | 84% | 83% | 82% | 82% |
| **SVM** | **86%** | **85%** | **84%** | **84%** |

> **SVM** was chosen as the best model because it achieves the highest **accuracy (86%)** and **F1-Score (84%)** across all sentiment classes.

---

## 🔎 Key Findings

| Indicator | Finding |
|-----------|---------|
| **Word Frequency** | "good", "great", "excellent" dominate positive reviews |
| **Negative Patterns** | "bad", "terrible", "poor" common in negative reviews |
| **Text Length** | Positive reviews tend to be 20% longer than negative |
| **Rating Correlation** | Strong correlation between text sentiment and star rating |
| **TF-IDF Bigrams** | Bigrams improve accuracy by 5% over unigrams |

---

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- **Sentiment Distribution** — Pie chart and bar plot
- **Rating Analysis** — Distribution by sentiment
- **Word Clouds** — Visual representation of frequent words
- **Review Length Analysis** — Character and word count distributions
- **Confusion Matrix** — Best model performance
- **Feature Importance** — Top TF-IDF features
- **Category Analysis** — Sentiment by product category

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud joblib
```

### 1. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. Run the Notebook

Open `sentiment_analysis.ipynb` in Jupyter Lab / VS Code for the complete step-by-step analysis with visualizations.

### 3. Use the Prediction Function

```python
from textblob import TextBlob

def predict_sentiment(review_text):
    # Preprocess and predict
    cleaned = preprocess_text(review_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return le.inverse_transform(prediction)[0]

# Example
predict_sentiment("This product is amazing! Best purchase ever!")
# Output: 'positive'
```

---

## 📁 Project Structure

```
sentiment-analysis/
├── sentiment_analysis.ipynb          # Jupyter notebook (full analysis)
├── requirements.txt                  # Python dependencies
├── models/
│   ├── sentiment_model.pkl          # Trained SVM model
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer
│   └── label_encoder.pkl            # Label encoder
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | Natural language processing |
| TextBlob | Text analysis |
| Scikit-learn | ML models & TF-IDF |
| WordCloud | Text visualization |
| Joblib | Model serialization |

---

**Built with Python, NLTK, Scikit-learn & WordCloud**

[![GitHub stars](https://img.shields.io/github/stars/basitali08/sentiment-analysis?style=social)](https://github.com/basitali08/sentiment-analysis) [![GitHub forks](https://img.shields.io/github/forks/basitali08/sentiment-analysis?style=social)](https://github.com/basitali08/sentiment-analysis)

---

**Built by Basit Ali** · [GitHub](https://github.com/basitali08) · [Email](mailto:whoisbasit@gmail.com)  
NLP & Text Analysis Machine Learning · MS Data Science Portfolio
