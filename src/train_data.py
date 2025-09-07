import pandas as pd
import numpy as py
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.logger import logger
from src.config import models,REPORTS_PATH,MODEL_SAVED_PATH
from src.customized_exceptions import ModelTrainingException
import joblib

def train_data(X_train_scaled,X_test_scaled,y_train,y_test):
    model_r2={}
    model_mse={}
    model_mae={}
    try:
        logger.info("Training has been started")
        for name,model in models.items():
            model.fit(X_train_scaled,y_train)
            y_predict=model.predict(X_test_scaled)
            model_r2[name]=r2_score(y_test,y_predict)
            model_mse[name]=mean_squared_error(y_test,y_predict)
            model_mae[name]=mean_absolute_error(y_test,y_predict)

        with open(REPORTS_PATH,"w") as file:
            name_of_model = max(model_r2, key=model_r2.get)
            R2_Value = model_r2[name_of_model]
            if name_of_model in model_mse and model_mae:
                mse_value=model_mse[name_of_model]
                mae_value=model_mae[name_of_model]
                content=f"Name of best model is{name_of_model}\n R2 score is {R2_Value}\n Mean Squared error is {mse_value}\n Mean absolute error is {mae_value}"
                file.write(content)
        if name_of_model in models:
            best_model=models[name_of_model]
            joblib.dump(best_model,MODEL_SAVED_PATH)
            
        logger.info("Training has been done and report has been generated")   
        return name_of_model 
   
    except Exception as e:
        logger.info("Training failed{e}")    
        raise ModelTrainingException(e)





