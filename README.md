
# AI Powered Toxic Comment Classification 

This is a **multi-label text classification** project that detects various types of toxicity in user-submitted comments. Built using **Streamlit** and trained with the **Jigsaw Toxic Comment Classification dataset**, the model predicts whether a comment is:

- **Toxic**
- **Severely Toxic**
- **Obscene**
- **Threatening**
- **Insulting**
- **Identity Hate**

## 📊 Demo

Check out the live app here 👉 [Toxic Comment Classifier on Streamlit Cloud]([https://flag-harmful-comment.streamlit.app)

![App Screenshot](./assets/demo_screenshot.png)
---

## 📁 Dataset

The app uses the **Jigsaw Toxic Comment Classification Challenge** dataset originally from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), containing over 150,000 Wikipedia comments labeled for different types of toxicity.

> 🔍 **Note**: In this project, the dataset was imported via [this GitHub repo](https://github.com/praj2408/Jigsaw-Toxic-Comment-Classification) which hosts a copy of the data.

---

## 🧠 Model Training

- **Vectorization**: TF-IDF
- **Classifier**: Logistic Regression
- **Preprocessing**:
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Tokenization

> 🗂️ A separate model was trained per toxicity label and a dictionary of the models was saved using `joblib`.

---

## 🚀 Running the App

To run the app locally:

```bash
# Clone the repo
git clone https://github.com/EbubeOkafor/toxic-comment-classifier.git
cd toxic-comment-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

📦 File Structure

.

├──Toxic-Comment-Classifier
├── app.py               # Streamlit web app

├── toxic_models.joblib/               # Saved trained models (.joblib)

|
├──vectorizer.pkl
├── requirements.txt     # Python dependencies

└── README.md            # This file


---

💡 Features

Accepts custom user comment as input

Predicts multiple toxicity labels simultaneously

Clean UI powered by Streamlit

Acknowledges that comments can belong to multiple classes



---

🛠️ Future Work

Improve preprocessing pipeline

Train with deep learning models (BERT, LSTM)



---
