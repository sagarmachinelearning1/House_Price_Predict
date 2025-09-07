import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE,RANDOM_STATE,SCALED_DATA_PATH
from src.logger import logger
from src.customized_exceptions import StandardizationException
from sklearn.preprocessing import StandardScaler
import joblib

def standardize_data(X_data,Y_data):
    try:
        logger.info("Standardization has been started")
        X_train,X_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        joblib.dump(scaler,SCALED_DATA_PATH)
        logger.info("Standardization has been done")
        return X_train_scaled,X_test_scaled,y_train,y_test
    
    except Exception as e:
        logger.error("standardization failed{e}")
        raise StandardizationException(e)



