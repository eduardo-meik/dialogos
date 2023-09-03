import streamlit as st
from transformers import pipeline, RobertaTokenizerFast, TFRobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

# Sentiment Analysis Pipeline
sentiment_pipe = pipeline('sentiment-analysis')

# Toxicity Classifier
model_path_toxic = "citizenlab/distilbert-base-multilingual-cased-toxicity"
toxicity_classifier = pipeline("text-classification", model=model_path_toxic, tokenizer=model_path_toxic)

# Emotion Analysis
tokenizer_emotion = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model_emotion = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model=model_emotion, tokenizer=tokenizer_emotion)

# User Needs Analysis
tokenizer_needs = AutoTokenizer.from_pretrained("thusken/nb-bert-base-user-needs")
model_needs = AutoModelForSequenceClassification.from_pretrained("thusken/nb-bert-base-user-needs")
user_needs = pipeline('text-classification', model=model_needs, tokenizer=tokenizer_needs)

st.title("Plataforma de Diálogos Participativos")

# Text area for input in sidebar
text = st.sidebar.text_area("Añade el texto a evaluar")

# Create columns for buttons in sidebar
col1, col2, col3, col4 = st.sidebar.columns(4)

# Place each button in a separate column
run_sentiment_analysis = col1.button("Evaluar Sentimiento")
run_toxicity_analysis = col2.button("Evaluar Toxicidad")
run_emotion_analysis = col3.button("Evaluar Emoción")
run_user_needs_analysis = col4.button("Evaluar Necesidades del Usuario")

# Container for output in main layout
output_container = st.container()

# Sentiment analysis
if run_sentiment_analysis and text:
    with output_container:
        sentiment_output = sentiment_pipe(text)
        label = sentiment_output[0]['label']
        score = round(sentiment_output[0]['score'] * 100, 2)
        st.markdown(f"**Resultado del análisis de sentimiento:**\n\n- **Etiqueta:** {label}\n- **Confianza:** {score}%")
elif run_sentiment_analysis and not text:
    st.sidebar.warning("Por favor, añade un texto para evaluar el sentimiento.")

# Toxicity analysis
if run_toxicity_analysis and text:
    with output_container:
        toxicity_output = toxicity_classifier(text)
        label = toxicity_output[0]['label']
        score = round(toxicity_output[0]['score'] * 100, 2)
        st.markdown(f"**Resultado del análisis de toxicidad:**\n\n- **Etiqueta:** {label}\n- **Confianza:** {score}%")
elif run_toxicity_analysis and not text:
    st.sidebar.warning("Por favor, añade un texto para evaluar la toxicidad.")

# Emotion analysis
if run_emotion_analysis and text:
    with output_container:
        emotion_output = emotion(text)
        label = emotion_output[0]['label']
        score = round(emotion_output[0]['score'] * 100, 2)
        st.markdown(f"**Resultado del análisis de emoción:**\n\n- **Etiqueta:** {label}\n- **Confianza:** {score}%")
elif run_emotion_analysis and not text:
    st.sidebar.warning("Por favor, añade un texto para evaluar la emoción.")

# User needs analysis
if run_user_needs_analysis and text:
    with output_container:
        needs_output = user_needs(text)
        label = needs_output[0]['label']
        score = round(needs_output[0]['score'] * 100, 2)
        st.markdown(f"**Resultado del análisis de necesidades del usuario:**\n\n- **Etiqueta:** {label}\n- **Confianza:** {score}%")
elif run_user_needs_analysis and not text:
    st.sidebar.warning("Por favor, añade un texto para evaluar las necesidades del usuario.")

