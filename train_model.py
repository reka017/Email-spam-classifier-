import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib


# 1. Load the dataset
data_path = "data/SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# 2. Convert labels into numbers
# ham = 0
# spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 3. Separate input and output
X = df["message"]
y = df["label"]

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# 6. Train the model
print("Training the Spam Classifier...")
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save the trained model
model_path = "models/spam_classifier.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully to: {model_path}")