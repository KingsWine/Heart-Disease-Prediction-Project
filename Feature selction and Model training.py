import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn import metrics

df = st.session_state['df']
df = df.drop(['education', 'ageGroup'], axis=1)

st.header('Feature Selection and Model Training')
st.write('This is to check what predictors are better than others using the Chi-Squared Test, Recursive Feature '
         'Elimination(RFE) and Decision Tree Method(Random Forest)')

# to clean and change the data types of some features
df.columns = [col.lower() for col in df.columns]
df = df.dropna()
df['cigsperday'] = df['cigsperday'].astype(int)
df['bpmeds'] = df['bpmeds'].astype(int)
df['totchol'] = df['totchol'].astype(int)
df['glucose'] = df['glucose'].astype(int)
df['heartrate'] = df['heartrate'].astype(int)
df['sysbp'] = df['sysbp'].astype(int)
df['diabp'] = df['diabp'].astype(int)
df['bmi'] = df['bmi'].astype(int)

# to split the predictor from the response
x = df.drop('tenyearchd', axis=1)
y = df['tenyearchd']

# feature selection with Chi-squared test
num_feats = 10  # to set the number of predictors
x_norm = MinMaxScaler().fit_transform(x)
chi_selector = SelectKBest(score_func=chi2, k=num_feats)
fit = chi_selector.fit_transform(x, y)
chi_support = chi_selector.get_support()
chi_feature = x.loc[:, chi_support].columns.tolist()

# feature selection with recursive feature selection
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(x, y)
rfe_support = rfe_selector.get_support()
rfe_feature = x.loc[:, rfe_support].columns.tolist()

# feature selection with embedded recursive feature selection
embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embedded_rf_selector.fit(x, y)
embedded_rf_support = embedded_rf_selector.get_support()
embedded_rf_feature = x.loc[:, embedded_rf_support].columns.tolist()

feature_name = list(x.columns)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature': feature_name, 'Chi-2': chi_support, 'RFE': rfe_support,
                                     'Random Forest': embedded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df) + 1)
st.write(feature_selection_df.head(15))

# to find and display the fisher score for each feature
st.subheader('Bar Chart to display the Fisher Score of the Predictors')
f_scores, p_values = f_classif(x, y)
features_rank = pd.DataFrame({'feature': df.columns[0:len(df.columns) - 1], 'fisherScore': f_scores, })
bar_chart = alt.Chart(features_rank).mark_bar().encode(y='fisherScore', x='feature')
st.altair_chart(bar_chart, use_container_width=True)

st.write(
    '**Considering the fisher score and the chi-squared method, the final features(predictors) selected for the model '
    'will be (including totChol because this variable gives total cholesterol levels and may be relevant):**')
st.write('1. prevalentHyp')
st.write('2. diabetes')
st.write(
    '3. BPMeds (whether the patient was on blood pressure medication - this indicates the patient is hypertensive.)')
st.write('4. age')
st.write('5. Gender')
st.write('6. sysBP (systolic blood pressure)')
st.write('7. prevalentstroke')
st.write('8. diaBP (diastolic blood pressure)')
st.write('9. cigsperday')
st.write('10. glucose')
st.write('11. totChol')

st.write('**Features to exclude:**')
st.write('1. BMI')
st.write('2. currentSmoker (redundant with cigsperday)')
st.write('3. heartRate (only one method chose this one)')

x = x.drop(['bmi', 'currentsmoker', 'heartrate'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

logr = LogisticRegression()
logr.fit(x_train, y_train)
y_pred = logr.predict(x_test)

st.session_state['logr'] = logr
st.session_state['y_test'] = y_test

cm = metrics.confusion_matrix(y_test, y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

st.write("**Classification Report**")

classification_report = {'accuracy': accuracy, 'precision': precision, 'recall_score': recall, 'f1_score': f1_score}
class_report = pd.DataFrame(classification_report, index=[0],
                            columns=['accuracy', 'precision', 'recall_score', 'f1_score'])
st.write(class_report)

# to create the confusion matrix
st.write("**Confusion Matrix**")
st.write(pd.DataFrame(cm, columns=["Predicted No CHD", "Predicted Possible CHD"],
                      index=["Actual No CHD", "Actual Possible CHD"]))

# display the confusion matrix graph
st.write("**Confusion Matrix Heatmap**")

heatmap_data = pd.DataFrame(cm, columns=["No CHD", "Possible CHD"],
                            index=["No CHD", "Possible CHD"]).stack().reset_index()
heatmap_data.columns = ["Actual", "Predicted", "Count"]

heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X('Predicted:N', title='Predicted'),
    y=alt.Y('Actual:N', title='Actual'),
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'))

text = heatmap.mark_text(baseline='middle').encode(
    text=alt.Text('Count:Q', format='d'),
    color=alt.condition(
        alt.datum.Count > cm.max() / 2,
        alt.value('white'),
        alt.value('black')
    )
)

heatmap_with_text = (heatmap + text).properties(width=500, height=400)

st.altair_chart(heatmap_with_text)
