import streamlit as st
import pickle

# ---------------- Load Pipeline ----------------
try:
    with open("spam_classifier_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Pipeline not found! Make sure 'spam_classifier_pipeline.pkl' is in the same folder.")
    st.stop()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="📩 Spam Classifier",
    page_icon="📨",
    layout="wide"
)

# ---------------- Title ----------------
st.markdown("""
<div style="text-align:center">
    <h1>📩 Spam Classifier</h1>
    <p style="font-size:16px;">Detect if a message is <b>Spam</b> or <b>Not Spam</b> with probability</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown("### ✏️ Enter your message below:")
input_sms = st.text_area(
    "",
    height=150,
    placeholder="Type your SMS or email message..."
)

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner('Analyzing message...'):
            result = pipeline.predict([input_sms])[0]
            prob = pipeline.predict_proba([input_sms])[0][1]

        st.markdown("---")

        # ---------------- Columns for Result ----------------
        col1, col2 = st.columns([1, 2])

        # Result card
        with col1:
            if result == 1:
                st.markdown(
                    f"""
                    <div style="background-color:#ff4b4b;padding:20px;border-radius:10px;text-align:center">
                        <h2 style="color:white">🚨 SPAM</h2>
                        <p style="color:white;font-size:18px;">Be careful! This message is likely spam.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#4bb543;padding:20px;border-radius:10px;text-align:center">
                        <h2 style="color:white">✔️ NOT SPAM</h2>
                        <p style="color:white;font-size:18px;">This message is safe.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

        # Probability progress
        with col2:
            st.markdown("### 📊 Spam Probability")
            st.progress(int(prob * 100))
            st.info(f"Probability: **{prob*100:.2f}%**")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
