import streamlit as st
import pandas as pd
import joblib
from src.predict_data import predict_data



st.title("Housing Price Predictor")


longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=37.0)
housing_median_age = st.slider("Housing Median Age", 1, 100, value=30)
population = st.number_input("Population", min_value=0, value=1000)
households = st.number_input("Households", min_value=0, value=300)
median_income = st.number_input("Median Income (in 10k USD)", min_value=0.0, value=5.0)
avg_bedrooms_per_household = st.number_input("Avg Bedrooms per Household", min_value=0.0, value=1.5)
avg_rooms_per_household = st.number_input("Avg Rooms per Household", min_value=0.0, value=4.5)


st.subheader("Ocean Proximity")
ocean_proximity_1H_OCEAN = st.number_input("1H OCEAN", min_value=0, max_value=1, value=0)
ocean_proximity_INLAND = st.number_input("INLAND", min_value=0, max_value=1, value=0)
ocean_proximity_ISLAND = st.number_input("ISLAND", min_value=0, max_value=1, value=0)
ocean_proximity_NEAR_BAY = st.number_input("NEAR BAY", min_value=0, max_value=1, value=0)
ocean_proximity_NEAR_OCEAN = st.number_input("NEAR OCEAN", min_value=0, max_value=1, value=0)


input_data = pd.DataFrame([{
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'population': population,
    'households': households,
    'median_income': median_income,
    'avg_bedrooms_per_household': avg_bedrooms_per_household,
    'avg_rooms_per_household': avg_rooms_per_household,
    'ocean_proximity_1H OCEAN': ocean_proximity_1H_OCEAN,
    'ocean_proximity_INLAND': ocean_proximity_INLAND,
    'ocean_proximity_ISLAND': ocean_proximity_ISLAND,
    'ocean_proximity_NEAR BAY': ocean_proximity_NEAR_BAY,
    'ocean_proximity_NEAR OCEAN': ocean_proximity_NEAR_OCEAN
}])


st.subheader("Model Input Data")

if st.button("Predict House Value"):
    prediction =predict_data(input_data)
    st.success(f"Predicted Median House Value: ${int(prediction):,}")
