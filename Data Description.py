import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.header('Data Description')

df = pd.read_csv("framingham.csv")  # importing the dataset

st.session_state['df'] = df

st.subheader('Header of Dataframe')
st.write(df.head())  # to display the first five rows of the dataset

st.subheader('Statistics of Dataframe')
st.write(df.describe().T)  # to display the descriptive statistics of the data

st.write('**Available null values per column**')
st.write(df.isnull().sum())  # to count the number of null values for each column
st.write('We see that out of 4240 rows 582, (14%)of the rows have missing values. Looking at each variable, '
         'the one with the highest percentage of missing values is glucose (9%). As all the variables with missing '
         'values have less than 25% of missing information, it will not be considered deleting the whole column but '
         'only the rows that contain missing information.')

# data cleaning - checking and removing nulls and duplicates

df = df.dropna()  # to drop the null values
df.columns = [col.lower() for col in df.columns]

st.write('**Dealing with missing data on the dataset:** Missing data can be treated by essentially two methods:')

st.write('Imputation')
st.write('Deleting')
