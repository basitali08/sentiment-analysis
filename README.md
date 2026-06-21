# Sentiment Analysis

## 📋 Project Overview

This project performs sentiment analysis on product reviews to classify them as positive, negative, or neutral. Using Natural Language Processing (NLP) techniques and machine learning, we build a model that can automatically analyze customer feedback.

## 🎯 Business Objective

- Automatically classify customer reviews by sentiment
- Understand customer satisfaction patterns
- Identify common themes in positive/negative reviews
- Build a scalable sentiment classification system

## 📊 Dataset

The dataset contains product reviews with:
- **Review Text**: Customer review content
- **Rating**: 1-5 star rating
- **Sentiment**: Positive/Negative/Neutral label
- **Product Category**: Type of product reviewed

## 🛠️ Technologies Used

- Python 3.8+
- Pandas & NumPy (Data Manipulation)
- NLTK & TextBlob (NLP)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Visualization)
- Jupyter Notebook

## 📁 Project Structure

```
Sentiment-Analysis/
├── README.md
├── requirements.txt
├── sentiment_analysis.ipynb
├── data/
│   └── product_reviews.csv
├── images/
│   └── (visualization outputs)
└── models/
    └── (saved models)
```

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('all')"
```

4. Run the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis.ipynb
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 78% | 76% | 75% | 75% |
| Logistic Regression | 84% | 83% | 82% | 82% |
| SVM | 86% | 85% | 84% | 84% |

## 📝 Key Findings

1. **Word Frequency**: "good", "great", "excellent" dominate positive reviews
2. **Negative Patterns**: "bad", "terrible", "poor" common in negative reviews
3. **Text Length**: Positive reviews tend to be longer
4. **Rating Correlation**: Strong correlation between text sentiment and star rating

## 👨‍💻 Author

Created as part of Data Science Project Portfolio

## 📄 License

This project is open source and available under the MIT License.
