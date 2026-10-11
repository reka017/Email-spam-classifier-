import streamlit as st
import joblib

# Load trained model
model = joblib.load("models/spam_classifier.pkl")

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Email Spam Classifier")
st.write("Enter an email or message below to check whether it is Spam or Ham.")

# Message input
message = st.text_area(
    "Enter your email/message:",
    height=180,
    placeholder="Type your email or message here..."
)

# Classify button
if st.button("🔍 Classify Message"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("🚨 SPAM")
            st.write("This message is likely to be spam.")

        else:
            st.success("✅ HAM")
            st.write("This message appears to be legitimate.")