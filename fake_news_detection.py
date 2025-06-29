import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load datasets
df_fake = pd.read_csv('data/Fake.csv')
df_real = pd.read_csv('data/True.csv')

# Add labels
df_fake['label'] = 0
df_real['label'] = 1

# Combine datasets
df = pd.concat([df_fake, df_real])[['text', 'label']]

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test_tfidf)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
