import numpy as np
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Inicializando o lemmatizer e o conjunto de stopwords
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

# Carregar o modelo treinado e o tokenizer
MODEL_PATH = 'lstm_binary_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Constante
MAX_SEQUENCE_LENGTH = 512 

# Funções necessárias

def lemmatize_word(word):
    """Lematiza uma palavra."""
    return lemmatizer.lemmatize(word)

def process_text(text):
    """Processa o texto."""
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    filtered_and_lemmatized_words = [lemmatize_word(word) for word in words if word.lower() not in english_stopwords]
    text = ' '.join(filtered_and_lemmatized_words)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def predict_sentiment(text):
    """Prevê o sentimento de um texto."""
    text = process_text(text)
    sequence_text = tokenizer.texts_to_sequences([text])
    X_text = pad_sequences(sequence_text, maxlen=MAX_SEQUENCE_LENGTH)
    y_pred = model.predict(X_text)
    sentiment = "Positive" if y_pred > 0.5 else "Negative"
    return sentiment


