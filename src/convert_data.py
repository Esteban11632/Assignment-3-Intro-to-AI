import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import os

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, '..', '..', '..', 'Datasets', 'vehicles_clean2.csv')
dataset_path = os.path.normpath(dataset_path)  # Clean up the path
data = pd.read_csv(dataset_path)

# Select the columns we want to use
data = data[["price", "year", "odometer"]]
labels = data["price"]
data = data.drop(columns="price")