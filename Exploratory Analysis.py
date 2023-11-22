import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px

df = st.session_state['df']

st.header('Exploratory Analysis')
st.write('In this section we will get to explore and know our dataframe by answering some important questions.')

# to create new categorical feature for age group
df['ageGroup'] = pd.cut(x=df['age'], bins=[30, 39, 49, 59, 70],
                        labels=['30-39 years', '40-49 years', '50-59 years', '60-70 years'])

# to create new dataframe from the existing one and re-map the values of some columns
data_cat = df.copy()
data_cat['gender'] = data_cat['gender'].map({0: 'female', 1: 'male'})
data_cat['tenyearchd'] = data_cat['TenYearCHD'].map({0: 'negative', 1: 'positive'})
data_cat['diabetes'] = data_cat['diabetes'].map({0: 'negative', 1: 'positive'})
data_cat['education'] = data_cat['education'].map({1: 'some high school or GED', 2: 'High School',
                                                   3: 'Some College or Vocational School', 4: 'College'})
data_cat['prevalentStroke'] = data_cat['prevalentStroke'].map({0: 'negative', 1: 'positive'})
data_cat['prevalentHyp'] = data_cat['prevalentHyp'].map({0: 'negative', 1: 'positive'})
data_cat['Count_10YCHD'] = 1

# to aggregate and display probability of chd by gender
st.subheader('How is the distribution of the heart disease probability by gender? Is educational level '
             'relevant?')
data_pv = data_cat.pivot_table('Count_10YCHD', ['gender', 'tenyearchd'], aggfunc="sum").reset_index()

st.write('**Ten Year CHD Probabilities by Gender**')
domain = ['negative', 'positive']
range_ = ['#85a3ff', '#2e5ce6']
c = alt.Chart(data_pv).mark_bar().encode(
    x='gender', y='Count_10YCHD',
    color=alt.Color('tenyearchd', scale=alt.Scale(domain=domain, range=range_)),
)
st.altair_chart(c, use_container_width=True)

# to aggregate and display probability of chd by education
st.write('**Ten Year CHD Probabilities by Education**')
data_pv1 = data_cat.pivot_table('Count_10YCHD', ['education', 'tenyearchd'], aggfunc="sum").reset_index()

domain = ['negative', 'positive']
range_ = ['#85a3ff', '#2e5ce6']
c = alt.Chart(data_pv1).mark_bar().encode(
    x='education', y='Count_10YCHD',
    color=alt.Color('tenyearchd', scale=alt.Scale(domain=domain, range=range_)),
)
st.altair_chart(c, use_container_width=True)

st.write('we must not forget that correlation is not causation but the population with lower education level is more '
         'prone to not having access to healthcare, because of their job situation. This may influence their chances '
         'of having CHD.')

# to aggregate and display the current smokers by gender and age group
st.subheader('Which gender and age group smokes more?')
st.write('**Number of Current Smokers by Gender and Age group**')

data_pv2 = data_cat.pivot_table('currentSmoker', ['gender', 'ageGroup'], aggfunc="sum").reset_index()

domain = ['male', 'female']
range_ = ['#2e5ce6', '#85a3ff']
c = alt.Chart(data_pv2).mark_bar().encode(
    x='ageGroup', y='currentSmoker',
    color=alt.Color('gender', scale=alt.Scale(domain=domain, range=range_)),
)

st.altair_chart(c, use_container_width=True)

st.write('In almost all age groups, men smoke more than women. We have a total of 4240 patients in the dataset, '
         'of which 2095 are smokers, this is 49% of the dataset.')

# to aggregate and visualize the relationship between CHD and diabetes, hypertension and stroke
st.subheader('Checking the relation between some health conditions (diabetes, hypertension, stroke) in the incidence '
             'of heart disease.')
option = st.selectbox('Pick an option', (data_cat.columns[6:9]))

st.write('You selected:', option)
data_pv3 = data_cat.pivot_table('Count_10YCHD', [option, 'tenyearchd'], aggfunc="sum").reset_index()

domain = ['positive', 'negative']
range_ = ['#2e5ce6', '#85a3ff']
c = alt.Chart(data_pv3).mark_bar().encode(
    x='tenyearchd', y='Count_10YCHD',
    color=alt.Color(option, scale=alt.Scale(domain=domain, range=range_)),
)

st.altair_chart(c, use_container_width=True)

# to aggregate and visualize the relationship between Age group and diabetes, hypertension, stroke, sysBP and diaBP
st.subheader('BMI, Cholesterol, Glucose, SystolicBP, DiastolicBP and Heart Rate levels by age group')

option = st.selectbox('Pick an option', ('totChol', 'BMI', 'heartRate', 'glucose', 'sysBP', 'diaBP'))

fig1 = px.box(data_cat, x="ageGroup", y=option)
st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

st.write('**From the box plots we can see that:**')

st.write('Those in age group 30-39 years seem to have lower levels of total cholesterol, with median of 210, '
         'the second age group on this list was 40 -49 years (227.5), 50-59 years (243), and 60 – 70 years (246) '
         'following the trend that total cholesterol levels tend to be higher in an older population.')

# Create a scatter plot matrix using Seaborn to check the correlation of continuous variables
st.subheader('Considering the continuous variables, can we set limits in each that indicate that the patient is more '
             'prone to develop heart disease?')

data_matrix = data_cat.drop(
    ['gender', 'education', 'ageGroup', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'currentSmoker',
     'tenyearchd', 'Count_10YCHD'], axis=1)

st.write('**Scatterplot Matrix (SPLOM) for Framingham Heart Study Dataset**')

sns.set(style="ticks")
scatter_plot = sns.pairplot(data_matrix, hue='TenYearCHD', palette={0: '#2e5ce6', 1: '#FF1919'}, markers="s")
scatter_plot._legend.set_title('TenYearCHD')
scatter_plot._legend.texts[0].set_text('Negative')
scatter_plot._legend.texts[1].set_text('Positive')

# Display the plot in Streamlit
st.pyplot(scatter_plot)
st.write('From the matrix plot above, we can see that for all continuous variables considered there is no clear limit.')
