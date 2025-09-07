import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH
from src.logger import logger
from src.load_data import load_data
from src.preprocess_data import preprocess_data
from src.feature_engineering import feature_engineering
from src.standardize_data import standardize_data
from src.train_data import train_data
from src.hyperparameter_tunning import hyperparameter_tunning

def main():
    logger.info("Data loading has been started")
    data=load_data(RAW_DATA_PATH)
    logger.info("Data has been loaded successfully")

    logger.info("Feature Engineering has started")
    feature_data=feature_engineering(data)
    logger.info("Feature Engineering has been done and graphs have been saved")
    
    logger.info("Data preprocessing has been started")
    X_data,Y_data=preprocess_data(feature_data)
    logger.info("Data Preprocessings has been done")
    
    logger.info("Standardization has been started")
    X_train_scaled,X_test_scaled,y_train,y_test=standardize_data(X_data,Y_data)
    logger.info("Standardizatio has been done")

    logger.info("Training has been started")
    best_model=train_data(X_train_scaled,X_test_scaled,y_train,y_test)
    logger.info(f"Training has been completed and best model is {best_model}")

    #logger.info("Hyper parameter tuniing started")
    #msg=hyperparameter_tunning(best_model,X_train_scaled,X_test_scaled,y_train,y_test)
    #logger.info(msg)

    print(X_data.columns)
    print(Y_data.columns)

if __name__=="__main__":
        main()


    

    



