
import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


nltk.download('movie_reviews')


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)


texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)


vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)


y_pred = classifier.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n--- Customer Review Sentiment Analysis ---")
while True:
    user_review = input("\nWrite a review (or type 'exit' to quit): ").strip()
    if user_review.lower() == 'exit':
        print("Exiting the sentiment analysis. Goodbye!")
        break

    try:
        user_review_vec = vectorizer.transform([user_review])
        sentiment = classifier.predict(user_review_vec)[0]
        proba = classifier.predict_proba(user_review_vec)[0]
        print(f"Sentiment: {'Positive' if sentiment == 'pos' else 'Negative'} "
              f"({max(proba)*100:.2f}% confidence)")
    except Exception as e:
        print(f"Error processing input: {e}")
