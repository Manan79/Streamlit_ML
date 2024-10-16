import streamlit as st
import joblib # pip install joblib
from sklearn.preprocessing import StandardScaler
st.header("Price Predictor")

val1= st.text_input("area")
val2= st.text_input("bedrooms")


import pandas as pd
df= pd.read_csv('Housing.csv')
# df.drop(labels=['Unnamed: 0','area','bedrooms'], axis=1, inplace=True)
scaler= StandardScaler()
scaler.fit(df.loc[:,['area','bedrooms']])




if st.button("Check"):
    val= scaler.transform([[val1,val2]])
    st.write(val)
    model= joblib.load('manan.pkl')
    answer = model.predict(val)
    st.subheader(f"Result is {answer[0]}")
    