import pandas as pd
import numpy as np
from src.config import models,REPORTS_PATH,MODEL_SAVED_PATH
from src.config import param_grids
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from src.logger import logger
from src.customized_exceptions import HyperparametertunningException
import joblib


def hyperparameter_tunning(best_model,X_train_scaled,X_test_scaled,y_train,y_test):
    try:
        logger.info("Hyperpaarameter tunning has been started")
        if best_model in models:
            model=models[best_model]
            grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grids[best_model],
                            cv=5,
                            scoring='r2',
                            n_jobs=-1)


        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        R2_score=r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae=mean_absolute_error(y_test,y_pred)
        joblib.dump(best_model,MODEL_SAVED_PATH)

        with open(REPORTS_PATH,"a") as file:
            content=f"Name of best model is{best_model}\n R2 score is {R2_score}\n Mean Squared error is {mse}\n Mean absolute error is {mae}"
            file.write(content)
        logger.info("Hyper parameter tunning has been done and best model {best_model} has been saved")
        return f"Hyper parameter tunning has been done and best model saved"
    
    except Exception as e:
        logger.error("Hyper parameter failed{e}")
        raise HyperparametertunningException(e)

    


        
        