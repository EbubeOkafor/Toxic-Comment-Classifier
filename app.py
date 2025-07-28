import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model(1).joblib')
vectorizer = joblib.load('vectorizer.pkl')

# Toxic comment categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Streamlit UI
st.title("Toxic Comment Classifier")
st.write("Enter a comment to check if it's toxic or safe.")

user_input = st.text_area("Comment", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Transform input
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)[0]

        # Display results
        st.subheader("Classification Results:")
        for i, label in enumerate(categories):
            if prediction[i]:
                st.error(f"⚠️ {label.upper()}")
        if prediction.sum() == 0:
            st.success("✅ Comment is clean.")
