import streamlit as st
import joblib

# Load model dictionary and vectorizer
models = joblib.load('toxic_model.joblib')  # This is a dict!
vectorizer = joblib.load('vectorizer.pkl')

# Labels from the model keys
categories = list(models.keys())

st.title("Toxic Comment Classifier")
st.write("Enter a comment to check if it's toxic or not in different categories.")

user_input = st.text_area("Comment", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Vectorize input once
        input_vec = vectorizer.transform([user_input])

        # Run prediction for each model
        predictions = {}
        for label, model in models.items():
            pred = model.predict(input_vec)[0]
            predictions[label] = pred

        # Collect toxic tags
        toxic_tags = [label.upper() for label, is_toxic in predictions.items() if is_toxic == 1]

        st.subheader("Classification Results:")
        if toxic_tags:
            if len(toxic_tags) == 1:
                st.error(f"⚠️ This comment is classified as: **{toxic_tags[0]}**")
            else:
                st.error("⚠️ This comment is toxic in multiple categories:")
                for tag in toxic_tags:
                    st.markdown(f"- ❗ **{tag}**")
        else:
            st.success("✅ This comment is clean.")
