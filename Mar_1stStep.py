import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()

def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

# Parameters
quantile = 0.9
n_samples = 10000
n_features = 30
n_miss_features = 5
interval_length = 100
np.random.seed(0)

# Generate X_i
X = np.random.normal(0, 100, (n_samples, n_features))
X1 = np.random.normal(1, 16, (n_samples, n_miss_features))
combined_X = np.concatenate((X, X1), axis=1)


# Generate Y_i using a linear relationship plus quantile-dependent noise
coefficients = np.random.normal(10, 400, n_features + n_miss_features)
lower_bound = -interval_length * quantile
higher_bound = interval_length + lower_bound
noise = np.random.normal(lower_bound, higher_bound, n_samples)

Y = np.dot(combined_X, coefficients) + noise

# Splitting the data into train, test, and validation sets
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

# First split to separate out the training set
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1 - train_ratio, random_state=0)

# Second split to separate out the test and validation sets
X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=validation_ratio/(test_ratio + validation_ratio), random_state=0)

# Adding constant for intercept
X_train_const = sm.add_constant(X_train)

# Train the quantile regression model
quantile_reg_model = sm.QuantReg(Y_train, X_train_const).fit(q=quantile)

# Adding constant to the test set for prediction
X_test_const = sm.add_constant(X_test)

# Predict quantiles for the test set
Y_pred_test = quantile_reg_model.predict(X_test_const)

# Calculate error E_i
E_i = Y_test - Y_pred_test

# Calculate the adjusted quantile value
adjusted_quantile = quantile * (1 + 1 / len(E_i))

# Calculate the empirical quantile of errors
Q_alpha_E = np.quantile(E_i, adjusted_quantile)

# Adding constant to the validation set for prediction
X_validation_const = sm.add_constant(X_validation)

# Predict quantiles for the validation set
Y_pred_validation = quantile_reg_model.predict(X_validation_const)

# Adjust the predictions with the empirical quantile of errors
Y_pred_validation_adjusted = Y_pred_validation + Q_alpha_E

# Calculate various error metrics for both unadjusted and adjusted predictions
# mae_unadjusted = mean_absolute_error(Y_validation, Y_pred_validation)
# mae_adjusted = mean_absolute_error(Y_validation, Y_pred_validation_adjusted)
mse_unadjusted = mean_squared_error(Y_validation, Y_pred_validation)
mse_adjusted = mean_squared_error(Y_validation, Y_pred_validation_adjusted)
quantile_loss_unadjusted = quantile_loss(Y_validation, Y_pred_validation, quantile)
quantile_loss_adjusted = quantile_loss(Y_validation, Y_pred_validation_adjusted, quantile)
# mbe_unadjusted = mean_bias_error(Y_validation, Y_pred_validation)
# mbe_adjusted = mean_bias_error(Y_validation, Y_pred_validation_adjusted)

# print("MAE Unadjusted:", mae_unadjusted, "MAE Adjusted:", mae_adjusted)
print("MSE Unadjusted:", mse_unadjusted, "MSE Adjusted:", mse_adjusted)
print("Quantile Loss Unadjusted:", quantile_loss_unadjusted, "Quantile Loss Adjusted:", quantile_loss_adjusted)
# print("MBE Unadjusted:", mbe_unadjusted, "MBE Adjusted:", mbe_adjusted)
print("Adjusted Quantile:", adjusted_quantile)

# Plotting the adjusted predictions
plt.figure(figsize=(10, 6))
plt.scatter(Y_validation, Y_pred_validation_adjusted, alpha=0.5)
plt.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'r--')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Actual vs Adjusted Predicted Y on Validation Set')
plt.show()

# Plotting the error E_i and marking the empirical quantile
plt.figure(figsize=(10, 6))
plt.hist(E_i, bins=30, density=True, alpha=0.6, color='g')
plt.axvline(Q_alpha_E, color='r', linestyle='dashed', linewidth=2)
plt.title('Distribution of Errors (E_i) with Empirical Quantile Marked')
plt.xlabel('Error (E_i)')
plt.ylabel('Density')
plt.show()
