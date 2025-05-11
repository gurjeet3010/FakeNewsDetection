# 📰 Fake News Detection using Machine Learning

This project implements a **Fake News Detection** system using machine learning and natural language processing techniques. It classifies news articles as either **Real** or **Fake** based on their textual content.

## 📌 Problem Statement

With the rapid spread of misinformation online, especially through social media, it's critical to have automated tools that can help identify fake news. This project addresses that by building a classification model trained on real-world datasets.

## ✅ Features

- Uses TF-IDF vectorization for text processing
- Combines both `title` and `text` for better accuracy
- Balanced dataset to avoid bias
- Achieves high accuracy with Naive Bayes classifier
- Easy-to-test custom news snippets

## 💡 Tech Stack

- Python 🐍
- pandas, scikit-learn
- TfidfVectorizer
- Multinomial Naive Bayes (for classification)

## 📂 Dataset

- `fake.csv` — Contains fake news articles
- `true.csv` — Contains real news articles

Both datasets should contain at least the following columns:
- `title`: Headline of the article
- `text`: Full text of the news article

## 🚀 How It Works

1. Load and clean the data
2. Combine `title` and `text` for context
3. Label data: `0` for Fake, `1` for Real
4. Balance the dataset
5. Vectorize text using TF-IDF
6. Train a Multinomial Naive Bayes classifier
7. Evaluate on test data
8. Test predictions on custom inputs

## 🧪 Sample Prediction

```python
sample_news = ["The President announced a new education policy for 2025."]
