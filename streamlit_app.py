#Importing libraries

import streamlit as st
import pandas as pd
import sklearn
import pickle



# Reading csv file

test_df = pd.read_csv("test_cleaned.csv")
train_df = pd.read_csv("train_cleaned.csv")

#Loading ML Model

with open('model.pkl', 'rb') as file:
  model = pickle.load(file)



#Setting Page Confirmation

st.set_page_config(page_title = "Titanic Survival", page_icon = "", layout ="centered")

st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'><h1 style='text-align:center; color:white;'>Titanic Predictor</h1></div>",unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; color:black;'>Find out if you would have survived during the Titanic disaster</h4>",unsafe_allow_html=True)

#Styling Streamlit Web App

col1, col2 = st.columns(2)

with col1:
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.image("titanic.jpg", use_column_width = True)

with col_2:
  p_class = st.selectbox(label = 'Select your economic class on the titanic', options = train_df['Pclass'].unique(), placeholder = 'Select your economic class on the titanic', index= none)

  gender = st.radio(label = 'Select your gender', options = train_df['Sex'].unique(), index= none)

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
    embarked = st.selectbox(label = 'Select your port of embarkation', options = train_df['Embarked'].unique(), placeholder = 'Select your port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', index= none)


  pred = st.button("Predict", use_container_width = True)

  #Creating DataFrame
  input_df = pd.DataFrame({'Pclass':[p_class], 'Sex':[gender], 'Age':[age], 'SibSp':[sibsp], 'Parch':[parch], 'Fare':[fare], 'Embarked':[embarked]})

  df1 = pd.DataFrame(input_df)


  #Defining the correct for Columns 

model_features = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked']

for feature in model_features: 
  if feature not in df1.columns: 
    df1[feature] = 0

df1 = df1[model_features]

#Making Prediction by Trained ML Model 

if pred: 
  if any([p_class is None, gender is None, age is None, sibsp is None, parch is None, fare is None, embarked is None]): 
    st.error("Please , select all inputs before pressing the predict button.", icon ="📝")
  else: 
    prediction = model.predict(df1)[0]
    if prediction < 0: 
      st.error('Prediction is below 0. Please select valid inputs',icon="⚠️")
    else: 
      st.success(f"Preidction of your survival is : {prediction}", icon = "✅")
