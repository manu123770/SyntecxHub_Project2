import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (run once)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data.csv")

# Prepare stopwords (KEEP "not")
stop_words = stopwords.words('english')
if 'not' in stop_words:
    stop_words.remove('not')

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
data['cleaned'] = data['text'].apply(clean_text)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned'])

# Target column
y = data['sentiment']

# Stratified split (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom input
while True:
    user_input = input("\nEnter a comment (or type 'exit'): ")

    if user_input.lower() == 'exit':
        break

    cleaned_input = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned_input])

    prediction = model.predict(vector_input)

    print("Prediction:", prediction[0])