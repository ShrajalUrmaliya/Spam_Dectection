import streamlit as st
import pickle

# ----------------------------
# 1️⃣ Load the pipeline
# ----------------------------
try:
    with open('spam_classifier_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("Pipeline not found. Make sure 'spam_classifier_pipeline.pkl' is in the same folder.")
    st.stop()

# ----------------------------
# 2️⃣ Streamlit UI
# ----------------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Email & SMS Spam Classifier")
st.write("Enter a message below and let the model classify it as Spam or Not Spam.")

input_sms = st.text_area("✏️ Enter your message here:", height=150, placeholder="Type your email or SMS message...")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please type a message before predicting.")
    else:
        with st.spinner("Analyzing message..."):
            # Pipeline handles preprocessing, TF-IDF, and prediction
            result = pipeline.predict([input_sms])[0]

            # Optional: probability if supported by model
            prob = None
            if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                prob = pipeline.predict_proba([input_sms])[0]

        # Display result
        if result == 1:
            st.error("🚨 **Spam Detected!**")
        else:
            st.success("✔️ **Not Spam**")

        if prob is not None:
            spam_prob = prob[1] * 100
            st.info(f"**Spam Probability:** {spam_prob:.2f}%")

st.caption("Made with ❤️ using Streamlit")
