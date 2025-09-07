import pandas as pd
import numpy as np
from src.logger import logger
from src.customized_exceptions import PreprocessingException

    
def preprocess_data(data):
        
        try:
            logger.info("Preprocessing has been started") 
    
            data=pd.get_dummies(data=data,columns=["ocean_proximity"]).astype(int)
            X_data=data.drop(columns="median_house_value",axis=1)
            Y_data=data[["median_house_value"]]
            X_data['avg_bedrooms_per_household'] = X_data['total_bedrooms'] / X_data['households']
            X_data['avg_rooms_per_household'] = X_data['total_rooms'] / X_data['households']
            X_data=X_data.drop(columns=["total_rooms","total_bedrooms"],axis=1)
            return X_data,Y_data
        
        except Exception as e:
             logger.info("Preprocessing failed")
             raise PreprocessingException(e)



