import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import CLEANSED_DATASET_PATH, NORNALIZED_DATASET_PATH
from sklearn.preprocessing import MinMaxScaler

def standardize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    return data

def correlation_matrix(data):
    correlation_matrix = data.corr()

    plt.imshow(correlation_matrix, cmap='Blues')
    plt.colorbar()
    
    variables = []
    for i in correlation_matrix.columns:
        variables.append(i)
    
    plt.xticks(range(len(correlation_matrix)), variables, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix)), variables)
    
    plt.show()
    

if __name__ == "__main__":
    
    data = pd.read_csv(NORNALIZED_DATASET_PATH)
    
    standardized_data = standardize_data(data)
    
    correlation_matrix(standardized_data)