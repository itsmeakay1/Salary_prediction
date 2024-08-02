import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction")

st.divider()

Years_at_Company = st.number_input("Enter year at company",min_value=0 ,max_value= 200)
Satisfaction_Level = st.number_input("Enter satisfaction level in the company",min_value=0)
Average_Monthly_Hours = st.number_input("Enter the average monthly hours",min_value=120)

x=[Years_at_Company,Satisfaction_Level,Average_Monthly_Hours]
sc=joblib.load("sc.pkl")
model=joblib.load("model.pkl")

predict_button = st.button("Press the button for the Predicted Salary")

st.divider()

if predict_button:

    st.balloons()

    x1=np.array(x)
    x_array = sc.transform([x1])
    prediction = model.predict(x_array)[0]
    st.write(f"salary prediction is {prediction}")
else:
    st.write("Please enter the vlaue for the prediction")





#Index(['Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours'], dtype='object')