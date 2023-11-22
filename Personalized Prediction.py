import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score

logr = st.session_state['logr']
y_test = st.session_state['y_test']

st.title("Personalized Prediction")
st.subheader('Welcome 😊')
name = st.text_input('Please enter name')

gender = st.radio("What's your gender?", ["Male", "Female"])
age = st.slider('How old are you?', 20, 80)
prevalentHbp = st.radio('Any prevalent case of High BP?', ['Yes', 'No'])
bpmeds = st.radio('Are you currently on BP medications?', ['Yes', 'No'])
diabetes = st.radio('Any case of Diabetes?', ['Yes', 'No'])
sysBP = st.slider('What is your Systolic BP?', 70, 300)
diaBP = st.slider('What is your Diastolic BP?', 40, 150)
prevalentStroke = st.radio('Any prevalent case of Stroke?', ['Yes', 'No'])
cigsperday = st.slider('How many sticks of cigarette do you smoke per day?', 0, 70)
glucose = st.slider('What is your glucose level?', 40, 400)
totchol = st.slider('What is your total cholesterol?', 100, 700)

gender = 1 if prevalentHbp == "male" else 0
prevalentHbp = 1 if prevalentHbp == "Yes" else 0
bpmeds = 1 if bpmeds == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0

entry = {'gender': gender, 'age': age, 'bpmeds': bpmeds, 'diabetes': diabetes, 'sysbp': sysBP, 'diabp': diaBP,
         'prevalentstroke': prevalentStroke, 'prevalentHbp': prevalentHbp, 'cigsperday': cigsperday, 'glucose': glucose,
         'totchol': totchol}
data = pd.DataFrame([entry], index=None)
st.write(data)



y_pred = logr.predict(data)
st.write(y_pred)

risk_prediction = logr.predict_proba(data)[:, 1]
st.write(risk_prediction)