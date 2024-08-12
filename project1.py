import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

# Step 1: Setup - Install necessary libraries (if not installed)
# !pip install nltk pandas scikit-learn

# Step 2: Data Preparation
nltk.download("movie_reviews")

# Load the dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Step 3: Model Training

# Convert text data to feature vectors
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 4: Prediction


def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]


# Test the prediction function
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
