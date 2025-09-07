import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logger
from src.customized_exceptions import FeatureEngineeringException
from src.config import GRAPHS_PATH
import os

def feature_engineering(data):
    try:
        
        logger.info("Feture Engineering has started")
        
        logger.info("Feature Selection has been started")
        data["total_bedrooms"]=data["total_bedrooms"].fillna(data["total_bedrooms"].mean())
        data["ocean_proximity"]=data["ocean_proximity"].replace("<1H OCEAN","1H OCEAN")
        data=data.drop_duplicates()
        cols=data.select_dtypes(include='number')
    

        new_columns=cols.columns
        for col in new_columns:

            plt.figure(figsize=(10,5))
            sns.scatterplot(x=cols[col],y=cols["median_house_value"])
            plt.show()
            
            plt.title(f"{col} vs Median House Value")

            os.makedirs(GRAPHS_PATH, exist_ok=True)

            filename = f"{col}_vs_median_house_value.png"
            filepath = os.path.join(GRAPHS_PATH, filename)
            plt.savefig(filepath)
            plt.close() 
        plt.figure(figsize=(10,5)) 
        sns.heatmap(cols.corr(),
        
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})   
        plt.title("Correlations graph")

        os.makedirs(GRAPHS_PATH, exist_ok=True)

        filename_correlation = f"Correlations graph"
        filepath_correlation = os.path.join(GRAPHS_PATH, filename_correlation)
        plt.savefig(filepath_correlation)
        plt.close() 
        
        
        ocean_proximity_group = data.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
        plt.figure(figsize=(10,5))
        sns.barplot(data=ocean_proximity_group,x="ocean_proximity",y="median_house_value")
        plt.title("Ocean Proximity vs Median House Value")
        filename_barplot="barplot"
        filepath_barplot=os.path.join(GRAPHS_PATH,filename_barplot)
        plt.savefig(filepath_barplot)
        plt.close()
        logger.info("Graphs has been saved")   
        return data 

    except Exception as e:
        logger.error(f"Feature engineering failed{e}")
        raise FeatureEngineeringException(e)




        