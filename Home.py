import streamlit as st
st.set_page_config(layout="wide")

st.title('CORONARY HEART DISEASE PREDICTION PROJECT')
st.image("heart.jpg", use_column_width=True)

st.write(
    " ***Disclaimer:** This application is intended for educational and informational purposes only."
    "\nThe insights and visualizations presented are based on historical data and associations observed in the "
    "Framingham Heart study.\n"
    "It is essential to consult qualified healthcare professionals for personalized medical assessments and advice.*\n")

st.write("**About the Project:**"
         "\n The goal of this project is to predict using Classification Machine learning model,"
         "\nwhether a patient has a risk of developing future Coronary Heart Disease.")

st.header("""Feature Description""")
st.write('The Framingham heart disease dataset includes over 4,240 records, 16 columns and 15 attributes.'
         '\nThis attributes include:')

st.subheader('Demographics:')
st.write('**Gender:** 0 for female, 1 for male (nominal)')
st.write('**Age:** the patient age (continuous)')

st.subheader('Behavioural:')
st.write('**Education level:** 1- Some high school or GED, 2- High school, 3- Some College or Vocational School, '
         '4- College (ordinal)')
st.write('**currentSmoker:** if the patient is a current smoker or not. 0 for No and 1 for yes (nominal)')
st.write('**cigsPerDay:** the number of cigarettes the patient smokes on an average in one day (continuous)')

st.subheader('Medical History:')
st.write('**BPMeds:** if the patient is on blood pressure medication or not. 0-No, 1-Yes (nominal)')
st.write('**prevalentStroke:** whether or not the patient had previously had a stroke. 0 - No, 1 - Yes (nominal)')
st.write('**prevalentHyp:** whether or not the patient was hypertensive. 0 - No, 1 - Yes (nominal)')
st.write('**diabetes:** whether or not the patient had diabetes. 0 - No, 1 - Yes (nominal)')

st.subheader('Medical - Current:')
st.write('**totChol:** the patients total cholesterol (continuous)')
st.write('**sysBP:** the patients systolic blood pressure (continuous)')
st.write('**diaBP:** the patients Diastolic blood pressure (continuous)')
st.write('**BMI:** the patients body mass index (continuous)')
st.write('**heartrate:** the patients present heart rate (continuous)')
st.write('**glucose:** the patients glucose level (continuous)')

st.subheader('Target Variable:')
st.write('**TenYearCHD:** 10 year risk of having a coronary heart disease (CHD). 0-No, 1-Yes (binary)')
