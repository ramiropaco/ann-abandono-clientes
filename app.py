import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from pathlib import Path

# Para ejecutar en la línea de comandos, ve al directorio y escribe: streamlit run app.py

# Obtener el directorio del script actual
BASE_DIR = Path(__file__).resolve().parent

# Cargar el modelo entrenado desde la misma carpeta
try:
    model = tf.keras.models.load_model(BASE_DIR / 'model.h5')
except FileNotFoundError:
    st.error(f"Error: No se encontró el modelo en: {BASE_DIR / 'model.h5'}.  Asegúrate de que el archivo 'model.h5' esté en el mismo directorio que este script.")
    st.stop()

# Cargar los codificadores y el escalador desde la misma carpeta
try:
    with open(BASE_DIR / 'label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open(BASE_DIR / 'onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open(BASE_DIR / 'scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error: No se pudo cargar un archivo necesario. Asegúrate de que todos los archivos .pkl estén en el mismo directorio que este script. Detalle del error: {e}")
    st.stop()
    
# Configuración de la aplicación Streamlit
st.title('Predicción de Fuga de Clientes')

# Entradas del usuario
geography = st.selectbox('Geografía', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Género', label_encoder_gender.classes_)
age = st.slider('Edad', 18, 92, step=1)
balance = st.number_input('Saldo', step=1, format="%d")
credit_score = st.number_input('Puntuación de Crédito', step=1, format="%d")
estimated_salary = st.number_input('Salario Estimado', step=1, format="%d")
tenure = st.slider('Antigüedad', 0, 10, step=1)
num_of_products = st.slider('Número de Productos', 1, 4, step=1)
has_cr_card = st.selectbox('Tarjeta de Crédito', [0, 1])
is_active_member = st.selectbox('Miembro Activo', [0, 1])

# Preparar los datos de entrada
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Codificar 'Geography' con one-hot encoding
# Convertir la entrada de geography en un DataFrame con el nombre de columna correcto
geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combinar columnas codificadas con los datos de entrada
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Realizar la predicción
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Convertir la probabilidad a porcentaje
porcentaje_abandono = prediction_proba * 100

# Mostrar resultados con formato de color según la predicción
if prediction_proba > 0.5:
    st.markdown(f'<h3 style="color:red">Probabilidad de abandono: {porcentaje_abandono:.2f}%</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:red; font-weight:bold">Es probable que el cliente abandone.</p>', unsafe_allow_html=True)
else:
    st.markdown(f'<h3 style="color:green">Probabilidad de abandono: {porcentaje_abandono:.2f}%</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:green; font-weight:bold">No es probable que el cliente abandone.</p>', unsafe_allow_html=True)

# Mostrar los datos del cliente
st.subheader("Datos del cliente:")
datos_cliente = {
    "Geografía": geography,
    "Género": gender,
    "Edad": int(age),
    "Antigüedad": int(tenure),
    "Saldo": int(balance),
    "Número de Productos": int(num_of_products),
    "Tarjeta de Crédito": "Sí" if has_cr_card == 1 else "No",
    "Miembro Activo": "Sí" if is_active_member == 1 else "No",
    "Puntuación de Crédito": int(credit_score),
    "Salario Estimado": int(estimated_salary)
}

# Convertir todos los valores a string para evitar problemas de serialización con Arrow
datos_cliente_str = {k: str(v) for k, v in datos_cliente.items()}

# Mostrar los datos en formato de tabla
st.table(pd.DataFrame([datos_cliente_str]).T.rename(columns={0: "Valor"}))