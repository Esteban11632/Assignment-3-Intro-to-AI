from convert_data import train_data, test_data, train_labels, test_labels
import numpy as np

# Insert a column of ones to serve as x0
train_data["ones"] = 1

# Select columns of interest
cols = train_data.columns
X = train_data[cols].values
Y = train_labels.values

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = np.transpose(X)
W = np.linalg.inv(XT @ X) @ XT @ Y

# Make predictions of price from test data
test_data["ones"] = 1
X_test = test_data[cols].values
pred_test = X_test @ W

# Make predictions of price from train data
X_train = train_data[cols].values
pred_train = X_train @ W

# Compute root mean-squared error of test data
error_test = pred_test - test_labels.values
rmse_test = (error_test **2).mean() ** .5
print("Test RMSE: {:.2f}".format(rmse_test))

# Compute root mean-squared error of train data
error_train = pred_train - train_labels.values
rmse_train = (error_train **2).mean() ** .5
print("Train RMSE: {:.2f}".format(rmse_train))

# Create a list of the abs(coeff) by feature
coeff_abs_list = []
for idx in range(len(W)):
    coeff_abs_list.append( (abs(W[idx]), cols[idx]) )
# Sort the list
coeff_abs_list.sort(reverse=True)

# Print the coefficients in order
for idx in range(len(W)):
    print("Feature: {:26s} abs(coef): {:.2f}".format(coeff_abs_list[idx][1], coeff_abs_list[idx][0]))