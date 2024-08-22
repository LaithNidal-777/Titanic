#Importing libraries

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

#Setting Page Cnfirmation

st.set_page_config(page_title = "Titanic Model", layout ="centered")

st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'><h1 style='text-align:center; color:white;'>Titanic Predictor</h1></div>",unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; color:black;'>Find out if you would have survived during the Titanic disaster</h4>",unsafe_allow_html=True)

# Reading csv file

test_df = pd.read_csv("test_cleaned.csv")
train_df = pd.read_csv("train_cleaned.csv")

#Styling Streamlit Web App

col1, col2 = st.columns(2)

with col1:
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.image("titanic.jpg", use_column_width = True)

with col2:
  p_class = st.selectbox(label = 'Select your economic class on the titanic', options = train_df['Pclass'].unique(), placeholder = 'Select your economic class on the titanic', index= None)

  gender = st.radio(label = 'Select your gender (0 for male 1 for female)', options = train_df['Sex'].unique(), index= None)

  age = st.number_input(label = 'Enter your age',placeholder="Enter your age",value=None,min_value=0,max_value=99,step=1)

  col3, col4 = st.columns(2)
  with col3:
    sibsp = st.number_input(label = 'Enter your number of siblings',placeholder="Enter your number of siblings",value=None,min_value=0,max_value=10,step=1)
  with col4:
    parch = st.number_input(label = 'Enter your number of parents and children accompanying you on the trip',placeholder="Enter your number of parents and children accompanying you on the trip",value=None,min_value=0,max_value=10,step=1)
  col5, col6 = st.columns(2)
  with col5:
    fare = st.number_input(label = 'Enter your fare',placeholder="Enter your fare ( Average Fare for pclass 1 : 84.15, pclass2:20.66, pclass3:13.68)",value=None,min_value=0,max_value=1000,step=10)
  with col6:
    embarked = st.selectbox(label = 'Select your port of embarkation 0 for Southampton, 1 for Cherbourg, 2 for Queenstown', options = train_df['Embarked'].unique(), placeholder = 'Select your port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', index= None)


  pred = st.button("Predict", use_container_width = True)


X_train = train_df.drop('Survived', axis = 1)#Features
y_train = train_df['Survived']#target

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 42)

#Feature Scaling 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test = scaler.transform(test_df)
#Selecting Model 

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Random Forest')
)

def add_parameter_ui(clf_name): 
  params = {}
  if clf_name == 'Logistic Regression': 
    C = st.sidebar.slider('C', 0.01, 10.0)
    params['C'] = C 
  else: 
    max_depth = st.sidebar.slider('max_depth', 2, 15)
    params['max_depth'] = max_depth
    n_estimators = st.sidebar.slider('n_estimators', 1, 100)
    params['n_estimators'] = n_estimators
  return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params): 
  clf = None
  if clf_name =='Logistic Regression': 
    clf = LogisticRegression(C = params['C'])
  else: 
    clf = RandomForestClassifier(n_estimators = params['n_estimators'], max_depth = params['max_depth'], random_state = 42)
  return clf

clf = get_classifier(classifier_name, params)
clf.fit(X_train, y_train)




  #Creating DataFrame
input_df = pd.DataFrame({'Pclass':[p_class], 'Sex':[gender], 'Age':[age], 'SibSp':[sibsp], 'Parch':[parch], 'Fare':[fare], 'Embarked':[embarked]})

df1 = pd.DataFrame(input_df)


  #Defining the correct for Columns 

model_features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']

for feature in model_features: 
  if feature not in df1.columns: 
    df1[feature] = 0

df1 = df1[model_features]

#Making Prediction by Trained ML Model 

if pred:

  if any([p_class is None, gender is None, age is None, sibsp is None, parch is None, fare is None, embarked is None]): 
    st.error("Please , select all inputs before pressing the predict button.", icon ="📝")
  else: 
    y_pred = clf.predict(df1)
    #acc = accuracy_score(y_val, y_pred)
    st.write(f'Survival Predition = {y_pred}')
    st.write(f'Classifier = {classifier_name}')
    #st.write(f'Accuracy = {acc}')
    if y_pred < 0: 
      st.error('Prediction is below 0. Please select valid inputs',icon="⚠️")
    else: 
      st.success(f"Preidction of your survival is : {y_pred}", icon = "✅")
