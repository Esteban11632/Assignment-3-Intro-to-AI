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

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(data, 
                                                                                            labels, 
                                                                                            test_size=0.2, 
                                                                                            shuffle=True, 
                                                                                            random_state=2025)

# Standardize the data
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds