import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

# Load model and preprocessing objects once at app start
@st.cache_resource
def load_model_and_objs():
    model = tf.keras.models.load_model("model_no_svd_clean.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_objs()

# Emoji helper based on predicted rating
def rating_emoji(rating):
    if rating >= 4.5:
        return "🥰"
    elif rating >= 3.5:
        return "🙂"
    elif rating >= 2.5:
        return "😐"
    elif rating >= 1.5:
        return "😕"
    else:
        return "😞"

# Sidebar info
st.sidebar.title("💄 About")
st.sidebar.write("""
Predict beauty product star ratings from user reviews and skin profile.
- Enter a single review or upload CSV for batch predictions.
- Ratings range from 1 to 5 stars.
""")

with st.sidebar.expander("💬 Example Reviews"):
    examples = {
        "🌟 5-star": "Absolutely love this! Made my skin feel so smooth and glowing.",
        "⭐ 1-star": "Terrible product. Caused irritation and did nothing it promised.",
        "⭐ 3-star": "It was okay. Not amazing but not bad either."
    }
    for label, text in examples.items():
        st.markdown(f"**{label}** — _{text}_")

# App Title
st.title("💄 Beauty Product Rating Predictor")
st.write("Predict star ratings from reviews + skin profile.")

# --- SINGLE REVIEW PREDICTION ---
with st.form("single_pred_form"):
    st.header("📝 Single Review Prediction")
    review_title = st.text_input("Review Title")
    review_text = st.text_area("Review Text")

    # Skin tone and skin type inputs
    skin_tone_grouped = st.selectbox("Skin Tone", ['deep', 'fair', 'fairLight', 'light', 'lightMedium', 'medium', 'mediumTan', 'rich', 'tan', 'other'])
    skin_type = st.selectbox("Skin Type", ['dry', 'oily', 'normal', 'combination'])

    submitted = st.form_submit_button("Predict Rating")

if submitted:
    if not review_title.strip() and not review_text.strip():
        st.error("Please enter a review title or review text.")
    else:
        with st.spinner("Predicting..."):
            full_review = review_title + " " + review_text
            input_df = pd.DataFrame([{
                "full_review": full_review,
                "skin_tone_grouped": skin_tone_grouped,
                "skin_type": skin_type,
                "is_recommended": "unknown"
            }])

            X_input_transformed = preprocessor.transform(input_df)
            X_tensor = tf.convert_to_tensor(X_input_transformed.toarray(), dtype=tf.float32)

            pred = model.predict(X_tensor).flatten()[0]
            pred = max(1.0, min(5.0, pred))
            pred_rounded = round(pred * 2) / 2
            emoji = rating_emoji(pred_rounded)
            recommended = "Yes" if pred_rounded >= 3.0 else "No"

            st.success(f"🌟 Predicted Rating: **{pred_rounded:.1f} stars** {emoji}")
            st.info(f"**Likely to Recommend?** {recommended}")
            st.write(f"*(Raw prediction: {pred:.3f})*")

            with open("prediction_log.csv", "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()},{skin_tone_grouped},{skin_type},{recommended},\"{full_review}\",{pred_rounded:.1f}\n")

st.markdown("---")

# --- BATCH PREDICTION ---
st.header("📅 Batch Prediction")
batch_file = st.file_uploader("Upload CSV with columns: review_title, review_text, skin_tone_grouped, skin_type", type="csv")

if batch_file is not None:
    try:
        df_batch = pd.read_csv(batch_file)
        required_cols = {'review_title', 'review_text', 'skin_tone_grouped', 'skin_type'}
        if not required_cols.issubset(df_batch.columns):
            st.error(f"CSV missing required columns: {required_cols}")
        else:
            st.info(f"Loaded {len(df_batch)} records for batch prediction.")
            df_batch['full_review'] = df_batch['review_title'].fillna('') + " " + df_batch['review_text'].fillna('')
            df_batch['is_recommended'] = 'unknown'
            df_input = df_batch[['full_review', 'skin_tone_grouped', 'skin_type', 'is_recommended']]

            X_input_transformed = preprocessor.transform(df_input)
            X_tensor = tf.convert_to_tensor(X_input_transformed.toarray(), dtype=tf.float32)

            with st.spinner("Running batch predictions..."):
                preds = model.predict(X_tensor).flatten()
                preds_clipped = [round(max(1.0, min(5.0, p)) * 2) / 2 for p in preds]

            df_batch['Predicted Rating'] = preds_clipped
            df_batch['Emoji'] = df_batch['Predicted Rating'].apply(rating_emoji)
            df_batch['Likely to Recommend'] = df_batch['Predicted Rating'].apply(lambda x: 'Yes' if x >= 3.0 else 'No')

            df_batch.rename(columns={"skin_tone_grouped": "skin_tone"}, inplace=True)

            st.write(f"### Batch Prediction Results ({len(df_batch)} samples)")
            st.dataframe(df_batch[['review_title', 'skin_tone', 'skin_type', 'Predicted Rating', 'Emoji', 'Likely to Recommend']])

            # Add Download Button
            csv = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="predicted_ratings.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("---")

# --- MODEL PERFORMANCE ---
st.header("📊 Model Performance on Test Set")

try:
    y_test = joblib.load("y_test.pkl")
    y_pred = joblib.load("y_pred_no_svd.pkl")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")

    if st.checkbox("Show Prediction vs. Actual Plot"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='Predictions')
        m, b = np.polyfit(y_test, y_pred, 1)
        ax.plot(y_test, m*y_test + b, color='red', linestyle='--', label='Fit line')
        ax.plot([1, 5], [1, 5], color='green', linestyle=':', label='Perfect prediction')
        ax.set_xlabel("Actual Ratings")
        ax.set_ylabel("Predicted Ratings")
        ax.set_title("Prediction vs. Actual Ratings (Test Set)")
        ax.legend()
        st.pyplot(fig)

    if st.checkbox("Show Residuals Histogram"):
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(residuals, bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Residual (Actual - Predicted)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residuals Distribution")
        st.pyplot(fig2)

except Exception as e:
    st.warning("Test set performance data not found or could not be loaded.")
    st.info("Please ensure 'y_test.pkl' and 'y_pred_no_svd.pkl' exist in the app directory.")
