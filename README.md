# 🩺 Clinical Trial Disease Category Classification

An end-to-end Machine Learning and NLP project that classifies clinical trial brief summaries into 8 disease categories using TF-IDF feature extraction and Logistic Regression, presented through an interactive Streamlit web dashboard.

---

## 📌 Project Overview
* **Dataset:** 60,337 clinical trial summaries sourced across 8 medical conditions.
* **Target Categories:** Breast Cancer, Type 2 Diabetes, COVID-19, Anxiety, Chronic Obstructive Pulmonary Disease (COPD), Rheumatoid Arthritis, Glaucoma, Sickle Cell Anemia.
* **Pipeline:** Regex Cleaning $\rightarrow$ Stopword Removal $\rightarrow$ WordNet Lemmatization $\rightarrow$ TF-IDF ($1, 2$ N-Grams) $\rightarrow$ Balanced Logistic Regression.
* **Performance:** ~94.7% Weighted F1-Score with calibrated class confidence probabilities.

---

## 📊 Dashboard Features
1. **🎯 Real-Time Disease Prediction:** Enter clinical summary text to generate the predicted medical category and a probability distribution bar chart.
2. **📊 In-Depth Clinical EDA:**
   - Class imbalance distribution and IQR word count boxplots.
   - Interactive marginal box histogram.
   - Category-wise lexical diversity metrics table.
   - Dynamic 2-word medical bigram extraction using CountVectorizer.

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/clinical-trial-nlp-classification.git
   cd clinical-trial-nlp-classification
