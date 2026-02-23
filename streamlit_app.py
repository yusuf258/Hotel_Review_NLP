import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Kütüphanesi Ayarları ---
# Hugging Face'de bazen NLTK data indirme sorunları olabiliyor, 
# bu yüzden her seferinde kontrol edip indiriyoruz.
nltk_packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
for package in nltk_packages:
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        try:
             nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package)

# --- Ön İşleme Fonksiyonu ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Sadece harfler ve boşluklar kalsın
    text = re.sub(r'[^a-z\s]', '', text)
    # Fazla boşlukları sil
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Model Yükleme (Cache Resource) ---
@st.cache_resource
def load_models():
    # BURASI ÇOK ÖNEMLİ:
    # Dosyanın (streamlit_app.py) bulunduğu klasörü bulur.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Modellerin kod ile AYNI KLASÖRDE (src içinde) olduğunu varsayıyoruz.
    model_path = os.path.join(current_dir, 'models', 'sentiment_model_dl.h5')
    tokenizer_path = os.path.join(current_dir, 'models', 'tokenizer_dl.pkl')
    
    # Hata ayıklama için (Eğer dosya bulunamazsa ekrana yazar)
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        return None, None
        
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

# Modelleri yükle
model, tokenizer = load_models()

# --- Uygulama Arayüzü ---
st.title("🏨 Hotel Review Sentiment Analysis")
st.markdown("**Sentiment analysis by Deep Learning  (LSTM)**.")

#st.info("Not: Bu proje Data Science eğitimi kapsamında geliştirilmiştir.")

user_input = st.text_area("Write the English comment you want to analyse below:", height=150)

if st.button("Analyse"):
    if user_input and model is not None:
        with st.spinner('Being analysed...'):
            # 1. Ön İşleme
            cleaned_input = clean_text(user_input)
            
            # 2. Tokenize & Padding (Eğitimdeki parametrelerle aynı olmalı: maxlen=150)
            sequence = tokenizer.texts_to_sequences([cleaned_input])
            padded_sequence = pad_sequences(sequence, maxlen=150)
            
            # 3. Tahmin
            prediction = model.predict(padded_sequence)
            sentiment_classes = ['Negative 😠', 'Neutral 😐', 'Positive 😃']
            
            # En yüksek olasılıklı sınıfı al
            class_index = np.argmax(prediction)
            predicted_sentiment = sentiment_classes[class_index]
            confidence = np.max(prediction) * 100
            
            # 4. Sonuç Gösterimi
            st.markdown("---")
            if class_index == 2: # Pozitif
                st.success(f"**Predicted sentiment:** {predicted_sentiment}")
            elif class_index == 0: # Negatif
                st.error(f"**Predicted sentiment:** {predicted_sentiment}")
            else: # Nötr
                st.warning(f"**Predicted sentiment:** {predicted_sentiment}")
                
            st.caption(f"Model Confidence Rate: %{confidence:.2f}")
            
    elif model is None:
        st.error("The process cannot be completed because the model could not be loaded.")
    else:
        st.warning("Please enter some text for analysis.")