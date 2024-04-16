import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()

# generate the data
# basic settings
quantile = 0.9
n_samples = 10000
n_X1 = 7
n_X2 = 2
n_X3 = 2
interval_length = 100
np.random.seed(0)

# generate data by GLM with misspecification
X1 = abs(np.random.normal(64, 100, (n_samples, n_X1)))
X2 = abs(np.random.normal(4, 10, (n_samples, n_X2)))
X3 = abs(np.random.normal(9, 10, (n_samples, n_X3)))

coefficients = abs(np.random.normal(10, 400, n_X1 + n_X2))
X = np.hstack((X1, X2, X3))
noise = np.random.normal(0, 16, n_samples)

X_true = X[:, :(n_X1 + n_X2)]
X_observed = np.hstack((X1, X3))
Y = np.dot(X_true, coefficients)
Y = Y**(-1/2) + noise

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2


X_train, X_temp, Y_train, Y_temp = train_test_split(X_observed, Y, test_size=1 - train_ratio, random_state=0)
X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=validation_ratio/(test_ratio + validation_ratio), random_state=0)

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

quantile_reg_model = sm.QuantReg(Y_train, X_train_const).fit(q=quantile)
linear_model = sm.OLS(Y_train, X_train_const).fit()
lasso = LassoCV(cv=5, random_state=0).fit(X_train_const, Y_train)



Y_pred_test = quantile_reg_model.predict(X_test_const)
Y_linear_pred_test = linear_model.predict(X_test_const)
Y_pred_validation_lasso = lasso.predict(X_test_const)

E_i = Y_test - Y_pred_test
E_i_linear = Y_test - Y_linear_pred_test
E_lasso = Y_test - Y_pred_validation_lasso

adjusted_quantile = quantile * (1 + 1 / len(E_i))
adjusted_quantile_linear = quantile*(1 + 1 / len(E_i_linear))
adjusted_quantile_lasso = quantile * (1 + 1 / len(E_lasso))

Q_alpha_E = np.quantile(E_i, adjusted_quantile)
Q_alpha_E_linear = np.quantile(E_i_linear, adjusted_quantile_linear)
Q_lasso = np.quantile(E_lasso, adjusted_quantile_lasso)

X_validation_const = sm.add_constant(X_validation)

Y_pred_validation = quantile_reg_model.predict(X_validation_const)
Y_pred_validation_adjusted = Y_pred_validation + Q_alpha_E
Y_pred_validation_linear = linear_model.predict(X_validation_const)
Y_pred_validation_linear_adjusted = Y_pred_validation_linear + Q_alpha_E_linear
Y_pred_validation_lasso = lasso.predict(X_validation_const)
Y_pred_validation_lasso_adjusted = Y_pred_validation_lasso + Q_lasso


quantile_loss_unadjusted = quantile_loss(Y_validation, Y_pred_validation, quantile)
quantile_loss_adjusted = quantile_loss(Y_validation, Y_pred_validation_adjusted, quantile)
quantile_loss_linear = quantile_loss(Y_validation, Y_pred_validation_linear, quantile)
quantile_loss_linear_adjusted = quantile_loss(Y_validation, Y_pred_validation_linear_adjusted, quantile)
quantile_loss_lasso = quantile_loss(Y_validation, Y_pred_validation_lasso, quantile)
quantile_loss_lasso_adjusted = quantile_loss(Y_validation, Y_pred_validation_lasso_adjusted, quantile)

print("Quantile Loss Unadjusted:", quantile_loss_unadjusted, "\n",  "Quantile Loss Adjusted:", quantile_loss_adjusted, "\n", 
      "Linear loss: ", quantile_loss_linear,  "\n","Linear adjusted loss:", quantile_loss_linear_adjusted, "\n",
      "Lasso loss: ", quantile_loss_lasso,  "\n","Lasso adjusted loss:", quantile_loss_lasso_adjusted)