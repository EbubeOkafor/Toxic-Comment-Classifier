import streamlit as st
import joblib
import string
import re

# Load model and vectorizer
model = joblib.load('toxic_model.joblib')
vectorizer = joblib.load('vectorizer.pkl')

# Toxic comment categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Basic stop words list (you can expand this or use NLTK)
stop_words = set([
    "a", "an", "the", "and", "is", "are", "was", "were", "in", "on", "at", "of", "for", "to", "from", "by", "with"
])

# Preprocessing function
def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

# Streamlit UI
st.title("Toxic Comment Classifier")
st.write("Enter a comment to check if it's toxic or safe.")

user_input = st.text_area("Comment", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Preprocess and transform input
        cleaned_text = preprocess(user_input)
        input_transformed = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_transformed)

        if hasattr(prediction, 'toarray'):
            prediction = prediction.toarray()

        prediction = prediction[0]

        # Display results
        st.subheader("Classification Results:")
        for i, label in enumerate(categories):
            if prediction[i]:
                st.error(f"⚠️ {label.upper()}")
        if prediction.sum() == 0:
            st.success("✅ Comment is clean.")
