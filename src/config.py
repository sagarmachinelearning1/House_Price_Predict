import os
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
RAW_DATA_PATH=os.path.join(BASE_DIR,"data","raw","house_price_predict.csv")
MODEL_SAVED_PATH=os.path.join(BASE_DIR,"models","best_model.pkl")
SCALED_DATA_PATH=os.path.join(BASE_DIR,"models","scaled.pkl")
LOG_FILE_PATH=os.path.join(BASE_DIR,"logs","records.log")
GRAPHS_PATH=os.path.join(BASE_DIR,"graphs")

TEST_SIZE=0.30
RANDOM_STATE=42

models={
    "Linear_Regression":LinearRegression(),
    "Ridge":Ridge(),
    "Lasso":Lasso(),
    "ElasticNet":ElasticNet(),
    "Decision_Tree_Regressor":DecisionTreeRegressor(),
    "Random_Forest_Regressor":RandomForestRegressor(),
    "Gradient_Boosting_Regressor":GradientBoostingRegressor(),
    "Xgboost":XGBRegressor(),
    "SVR":SVR(),
    "K-Neighbors_Regressor":KNeighborsRegressor()

}

param_grids = {
    'Random_Forest_Regressor': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'Gradient_Boosting_Regressor': {
        'n_estimators': [100, 200,300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0],
        'max_features': ['auto', 'sqrt']
    },
    'Xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10,20,50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    },
    'K-Neighbors_Regressor': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30, 40],
        'p': [1, 2] 
    }
}