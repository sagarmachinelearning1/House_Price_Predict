import pandas as pd
import numpy as np
import joblib
from src.config import MODEL_SAVED_PATH
from src.logger import logger
from src.customized_exceptions import PredictionException

def predict_data(input_data):
    try:
        logger.info("Prediction started")
        model=joblib.load(MODEL_SAVED_PATH)
        model.predict(input_data)
        prediction = model.predict(input_data)[0]
        logger.info("Prediction done")
        return prediction
    
    except Exception as e:
        logger.info("Prediction failed")
        raise PredictionException(e)


