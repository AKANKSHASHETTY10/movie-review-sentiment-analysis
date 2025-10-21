
🎬 Customer Review Sentiment Analysis

This project is a simple and fun way to explore sentiment analysis— figuring out whether a review sounds positive or negative.
It uses real movie reviews from NLTK, processes them using TF-IDF, and predicts emotions with a Naive Bayes classifier.

At the end, you can even type your own reviews and instantly see what the model thinks!

---

🌟 What This Project Does

* Reads thousands of movie reviews (positive and negative)
* Learns patterns using Machine Learning (Naive Bayes)
* Evaluates accuracy and prints a detailed report
* Lets you test it by typing your own review (like “The movie was amazing!”)
* Tells you how confident it is in its prediction

---

🧰 Tools & Libraries Used

* Python 3
* NLTK – for the movie review dataset
* scikit-learn – for training and evaluating the model
* TF-IDF Vectorizer – to convert text into numbers the model can understand

---

⚙️ How to Run It

1. Clone the Project

```bash
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
```

 2. Install the Required Packages

```bash
pip install nltk scikit-learn
```

3. Download the Movie Reviews Dataset

Open Python and run:

```python
import nltk
nltk.download('movie_reviews')
```

 4. Run the Script

```bash
python sentiment_analysis.py
```

🧩 How It Works (in Simple Terms)

1. The program reads movie reviews from NLTK.
2. It splits the data into training and testing parts.
3. Each review is turned into numbers using TF-IDF.
4. A Naive Bayes model learns which words usually mean positive or negative.
5. It tests its performance and prints accuracy.
6. Finally, you can type your own reviews and get instant results!


🚀 Future Improvements

* Add a simple web app using Streamlit or Flask
* Include more emotion categories (neutral, mixed feelings, etc.)
* Try deep learning models for better accuracy
* Add a dataset for product reviews or social media comments

