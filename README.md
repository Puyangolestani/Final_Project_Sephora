# Final_Project_Sephora

# From Emotion to Prediction: Using Data to Bridge the Beauty Gap

Welcome to the GitHub repository for my final project at the **WBS Coding School Data Science Bootcamp**. This project explores how machine learning and natural language processing (NLP) can be used to understand and predict consumer sentiment in the beauty industry — moving beyond generic review analysis to a more inclusive, predictive, and human-centered approach.

---

## 🧠 Project Overview

**"From Emotion to Prediction: Using Data to Bridge the Beauty Gap"** is a data science project focused on transforming subjective beauty product reviews into meaningful, actionable insights.

The core goal was to develop and deploy a machine learning model that predicts product ratings based on the emotional and contextual content of user reviews — including important demographic features like **skin tone** and **skin type**.

---

## 🎯 Project Objectives

- Predict user satisfaction (product rating) from text reviews.
- Understand emotional tone through sentiment analysis.
- Include often-overlooked features like **skin tone** and **skin type** for more inclusive predictions.
- Deploy the trained model as an interactive web app using **Streamlit**.

---

## 🔧 Technical Approach

### 1. Data Collection & Preprocessing
- Collected and cleaned over **40,000 beauty product reviews**.
- Extracted metadata, such as user demographics (skin tone, skin type).
- Applied text preprocessing (tokenization, stop word removal, vectorization).

### 2. Sentiment Analysis
- Applied NLP techniques to identify emotional signals.
- Used polarity and subjectivity metrics to enrich the feature set.

### 3. Modeling
- Compared **Random Forest** and **Deep Neural Network** models.
- Selected Deep Learning for better accuracy in predicting review scores.
- Evaluated model performance on validation sets.

### 4. Deployment with Streamlit
- Built an **interactive Streamlit web app** to demonstrate predictions.
- Users can input a review, select their skin tone/type, and get a real-time predicted rating.

---

## 🚀 Streamlit App

The live Streamlit app allows users to:
- Enter a beauty product review.
- Select their **skin tone** and **skin type**.
- Get a real-time prediction of the product rating based on the review content.

> **To run locally:**

```bash
streamlit run app.py
