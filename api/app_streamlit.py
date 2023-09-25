import streamlit as st
import requests
import time

# URL da sua API Flask
API_URL = "http://127.0.0.1:5000/predict"

#Image
#st.image('\\wsl.localhost\Ubuntu\home\edilson07\projects\sentiment_analysis\api\sentiment_robot.png', width=100)


st.title("Sentiment Analysis Prediction")

# Campo de texto para o usuário inserir a revisão
user_input = st.text_area("Enter your review:")

# Botão para fazer a previsão
if st.button("Predict"):
    # Enviar a revisão para a API Flask e obter a previsão
    response = requests.post(API_URL, json={"text": user_input})
    with st.spinner(text="Predicting..."):
        time.sleep(2)    
    
    # Verifique se a resposta foi bem-sucedida
    if response.status_code == 200:
        json_response = response.json()
        #st.write(f"Response from API: {json_response}")  # Escreva a resposta completa para depuração
        
        # Tente obter a previsão
        try:
            prediction = json_response["sentiment"]
            if prediction == "Positive":
                st.markdown("**Sentiment:** :smile: Positive!")
            else:
                st.markdown("**Sentiment:** :disappointed: Negative!")
        except KeyError:
            st.write("Key 'sentiment' not found in the API response.")
    else:
        st.write(f"Error: Received status code {response.status_code} from API.")



