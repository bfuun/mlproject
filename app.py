import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
st.set_page_config(page_title = "Salary Satisfaction App", layout = "wide")
st.title(" Salary Estimation App ")
st.markdown("#### Predict your expected salary based on company experience!")

st.image(r'\C:\Users\Admin\OneDrive\Pictures\woman-falling-money-5370748.webp', caption = "Let's Predict", use_container_width=True)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input("Years at Company ", min_value = 0, max_value = 20, value = 6)

with col2:
    satisfaction_level = st.slider("Satisfaction level ", min_value = 0.0, max_value = 1.0, step = 0.01, value = 0.7)
    
with col3:
    average_monthly_hours = st.slider("Average Monthly Hours ", min_value = 120, max_value = 310, step = 1,value = 6)
     
X = [years_at_company, satisfaction_level, average_monthly_hours]

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button(" Predict Salary ")

st.divider()

if predict_button:
    st.balloons()
    
    X_array = scaler.transform([np.array(X)])
    prediction = model.predict(X_array)
    
    st.success(f"Predicted Salary : ${prediction[0]:,.2f}")
    
    df_layout = pd.DataFrame({
        "Feature": ["Years at company", "Satisfaction Level", "Average Monthly Hours"],
        "Value": X
    })
    
    fig = px.bar(df_layout, x = "Feature", y = "Value", color = "Feature", title = " Your Input  Profile")
    st.plotly_chart(fig, use_container_width = True)
 
else:
    st.info(" Enter the details and press the Predict Salary button")
