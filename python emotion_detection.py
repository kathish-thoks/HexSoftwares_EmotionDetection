import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data silently
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Expanded dataset
texts = [
    # Happy
    "I am so happy and excited!",
    "I am extremely happy today!",
    "This is so funny, I can't stop laughing.",
    "That was hilarious, I laughed a lot.",
    "Feeling joyful and full of energy.",
    "I am thrilled about my new job.",
    
    # Sad
    "I feel sad and down.",
    "I feel really depressed and lonely.",
    "I am heartbroken and upset.",
    "I am crying because I miss my family.",
    "I feel empty and hopeless.",
    "Nothing makes me smile anymore.",
    
    # Angry
    "You make me angry!",
    "I am furious about this mistake.",
    "I am so mad right now.",
    "This is unacceptable and infuriating.",
    "I can't believe how rude that was.",
    "That made my blood boil.",
    
    # Fear
    "I am scared of the dark.",
    "That movie was scary!",
    "I feel nervous and anxious.",
    "I am terrified of spiders.",
    "My hands are shaking with fear.",
    "This situation is frightening.",
    
    # Neutral
    "I'm feeling peaceful and calm.",
    "I am relaxed and at ease.",
    "It's a normal day for me.",
    "I feel okay, nothing special.",
    "Just another ordinary day.",
    "Everything seems fine and steady."
]

emotions = [
    "happy", "happy", "happy", "happy", "happy", "happy",
    "sad", "sad", "sad", "sad", "sad", "sad",
    "angry", "angry", "angry", "angry", "angry", "angry",
    "fear", "fear", "fear", "fear", "fear", "fear",
    "neutral", "neutral", "neutral", "neutral", "neutral", "neutral"
]

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove punctuation/numbers
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

clean_texts = [preprocess(t) for t in texts]

# Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(clean_texts)
y = emotions

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Prediction function
def predict_emotion(text):
    text_clean = preprocess(text)
    text_vec = vectorizer.transform([text_clean])
    return model.predict(text_vec)[0]

# Test prediction
print("\n🎯 Sample Prediction:", predict_emotion("I am nervous about my exam."))

