# California House Price Prediction

This project predicts median house values in California using machine learning. It includes data preprocessing, feature engineering, training an XGBoost regression model, and hyperparameter tuning. The model is deployed via a Streamlit app for interactive predictions.

## Features Used
- Geographic: longitude, latitude
- Demographic: population, households, housing_median_age
- Economic: median_income
- One-hot encoded ocean proximity categories
- Engineered features: avg_bedrooms_per_household, avg_rooms_per_household


## Model
- Algorithm: XGBoostRegressor
- Typical performance: R² ≈ 0.74, MAE ≈ 40,000
- Hyperparameter tuning implemented to improve accuracy

## Output
- Predicts median house price based on user inputs via Streamlit interface.

> Note: This repository is a demonstration project intended for educational purposes and portfolio showcase. It is not designed for production use.



