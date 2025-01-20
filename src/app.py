import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)

print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())