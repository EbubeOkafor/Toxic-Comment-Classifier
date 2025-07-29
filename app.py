import streamlit as st
import joblib
import string
import re

# Load single-label model and vectorizer
model = joblib.load('toxic_model.joblib')         # This is for one label
vectorizer = joblib.load('vectorizer.pkl')        # TF-IDF vectorizer

# Target label name (update this if you used a different one)
target_label = 'identity_hate'

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = {"a", "an", "the", "and", "is", "are", "was", "were", "in", "on", "at", "of", "for", "to", "from", "by", "with"}
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("Toxic Comment Classifier")
st.write(f"Predicts whether a comment is '{target_label}' or clean.")

user_input = st.text_area("Comment", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned_text = preprocess(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        st.subheader("Classification Result:")
        if prediction:
            st.error(f"⚠️ This comment is classified as **{target_label.upper()}**")
        else:
            st.success("✅ This comment is **clean**.")
