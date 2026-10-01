import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial Disease Category Classifier",
    page_icon="🩺",
    layout="wide"
)

# Download NLTK data resources quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('clinical_model_lr.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return model, vectorizer, encoder, stop_words, lemmatizer

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_clinical_data.csv')
    if 'clean_word_count' not in df.columns:
        df['clean_word_count'] = df['cleaned_summary'].astype(str).apply(lambda x: len(x.split()))
    return df

@st.cache_data
def compute_lexical_table(df):
    records = []
    for cat in df['disease_category'].unique():
        tokens = ' '.join(df[df['disease_category'] == cat]['cleaned_summary'].astype(str)).split()
        tot = len(tokens)
        uniq = len(set(tokens))
        div = round((uniq / tot) * 100, 2) if tot > 0 else 0
        records.append({
            'Disease Category': cat.title(),
            'Total Cleaned Words': tot,
            'Unique Vocabulary': uniq,
            'Lexical Diversity (%)': div
        })
    return pd.DataFrame(records).sort_values(by='Total Cleaned Words', ascending=False)

@st.cache_data
def extract_top_bigrams(corpus_series, n=10):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    bag = vec.fit_transform(corpus_series.astype(str))
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:n], columns=['Bigram', 'Frequency'])

model, vectorizer, encoder, stop_words, lemmatizer = load_artifacts()
data = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ---------------------------------------------------------
# App Interface
# ---------------------------------------------------------
st.title("🩺 Clinical Trial Classification & Exploration Dashboard")
st.markdown("Automated disease category prediction and notebook-aligned exploratory data analysis.")

tab_pred, tab_eda = st.tabs([
    "🎯 Disease Prediction", 
    "📊 Clinical EDA"
])

# ---------------------------------------------------------
# Tab 1: Live Prediction
# ---------------------------------------------------------
with tab_pred:
    st.subheader("Predict Disease Category from Clinical Summary")
    
    sample_text = (
        "Phase 1 dose escalation and expansion study of adjuvant CDK4/6 inhibitors "
        "combined with endocrine therapy versus chemotherapy in women with hormone receptor-positive "
        "metastatic breast carcinoma."
    )
    
    user_input = st.text_area(
        "Enter Clinical Trial 'Brief Summary':",
        value=sample_text,
        height=140
    )
    
    if st.button("Classify Clinical Trial", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter medical summary text.")
        else:
            with st.spinner("Classifying clinical text..."):
                cleaned = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned])
                
                prediction_idx = model.predict(vectorized)[0]
                predicted_label = encoder.inverse_transform([prediction_idx])[0]
                probabilities = model.predict_proba(vectorized)[0]
                top_confidence = probabilities[prediction_idx] * 100
                
            st.success(f"**Predicted Category:** {predicted_label.title()}")
            st.metric("Model Confidence", f"{top_confidence:.2f}%")
            
            prob_df = pd.DataFrame({
                'Disease Category': [cls.title() for cls in encoder.classes_],
                'Confidence (%)': probabilities * 100
            }).sort_values(by='Confidence (%)', ascending=True)
            
            fig = px.bar(
                prob_df,
                x='Confidence (%)',
                y='Disease Category',
                orientation='h',
                title="Class Probability Distribution",
                text_auto='.1f'
            )
            fig.update_layout(xaxis_range=[0, 100], height=380)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Notebook-Aligned Clinical EDA
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    # 1. Headline Statistics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Studies", f"{len(data):,}")
    m2.metric("Mean Cleaned Words", f"{data['clean_word_count'].mean():.1f}")
    m3.metric("Median Words (50%)", f"{int(data['clean_word_count'].median())}")
    m4.metric("Max Word Outlier", f"{data['clean_word_count'].max()}")

    st.divider()

    # 2. Target Distribution & Cleaned Word Count (Box Plot)
    col_dist, col_box = st.columns(2)
    
    with col_dist:
        cat_counts = data['disease_category'].value_counts().reset_index()
        cat_counts.columns = ['Disease Category', 'Study Count']
        fig_dist = px.bar(
            cat_counts,
            x='Study Count',
            y='Disease Category',
            orientation='h',
            title="Class Imbalance: Study Counts Across 8 Diseases",
            text='Study Count',
            color='Study Count',
            color_continuous_scale='Teal'
        ).update_layout(yaxis={'categoryorder': 'total ascending'}, height=420, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_box:
        fig_box = px.box(
            data,
            x='disease_category',
            y='clean_word_count',
            color='disease_category',
            title="Cleaned Summary Word Count Across Diseases",
            labels={'clean_word_count': 'Word Count', 'disease_category': 'Category'}
        ).update_layout(height=420, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # 3. Interactive Plotly Histogram with Marginal Boxplot (Direct from Notebook)
    st.markdown("#### 🔬 Word Count Distribution & Outlier Spread")
    fig_plotly = px.histogram(
        data, 
        x='clean_word_count', 
        color='disease_category', 
        nbins=60,
        marginal='box', 
        title='Interactive Cleaned Word Count Distribution by Disease Category',
        labels={'clean_word_count': 'Cleaned Word Count', 'disease_category': 'Category'}
    )
    fig_plotly.update_layout(bargap=0.1, height=450)
    st.plotly_chart(fig_plotly, use_container_width=True)

    st.divider()

    # 4. Lexical Metrics Table & Top Bigrams Side-by-Side
    col_table, col_ngram = st.columns([1, 1.2])

    with col_table:
        st.markdown("#### 📚 Category-wise Lexical Diversity")
        st.caption("Lower diversity (%) reflects large corpora with repetitive terms; higher diversity reflects focused specialty cohorts.")
        lex_df = compute_lexical_table(data)
        st.dataframe(
            lex_df.style.format({
                'Total Cleaned Words': '{:,}',
                'Unique Vocabulary': '{:,}',
                'Lexical Diversity (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    with col_ngram:
        st.markdown("#### 🧬 Top Medical Bigrams (2-Word Keyphrases)")
        selected_cat = st.selectbox(
            "Select Disease Category to Inspect Bigrams:",
            sorted(data['disease_category'].unique().tolist())
        )
        
        subset_texts = data[data['disease_category'] == selected_cat]['cleaned_summary']
        top_bi_df = extract_top_bigrams(subset_texts, n=8).sort_values(by='Frequency', ascending=True)

        fig_bi = px.bar(
            top_bi_df,
            x='Frequency',
            y='Bigram',
            orientation='h',
            title=f"Signature Bigrams: {selected_cat.title()}",
            text='Frequency',
            color='Frequency',
            color_continuous_scale='Viridis'
        ).update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_bi, use_container_width=True)