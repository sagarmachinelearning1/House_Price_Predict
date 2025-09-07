import pandas as pd
import numpy as np
from src.logger import logger
from src.customized_exceptions import DataLoadException

def load_data(RAW_DATA_PATH):
    try:
        logger.info("Data loading started")
        data=pd.read_csv(RAW_DATA_PATH)
        logger.info("Data has been loaded")
        return data
    except Exception as e:
        logger.error("data loading failed{e}")
        raise DataLoadException(e)
        

   