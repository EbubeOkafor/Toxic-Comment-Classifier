import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('toxic_models.joblib')
vectorizer = joblib.load('vectorizer.pkl')

# Toxic comment categories (for multi-label classification)
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
        prediction = model.predict(input_transformed)

        if hasattr(prediction, 'toarray'):
            prediction = prediction.toarray()

        prediction = prediction[0]

        # Map prediction to category
        results = {label: int(prediction[i]) for i, label in enumerate(categories)}
        
        # Display results
        st.subheader("Classification Results:")
        toxic_tags = [label.upper() for label, is_toxic in results.items() if is_toxic]

        if toxic_tags:
            if len(toxic_tags) == 1:
                st.error(f"⚠️ This comment is classified as: **{toxic_tags[0]}**")
            else:
                st.error("⚠️ This comment is toxic in multiple categories:")
                for tag in toxic_tags:
                    st.markdown(f"- ❗ **{tag}**")
                st.info(f"Detected {len(toxic_tags)} toxic categories.")
        else:
            st.success("✅ This comment is clean.")
