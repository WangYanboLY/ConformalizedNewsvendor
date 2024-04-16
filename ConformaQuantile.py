import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers

# Neural Network for Regression
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def kernel_function(x1, x2):
    sigma = 1  
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

# def predict_kernel_quantile(X, Y, X_new, quantile):
#     n = len(X)
#     kernel_weights = np.array([kernel_function(X_new, X[i]) for i in range(n)])
#     sorted_Y = np.sort(Y)

#     for q in sorted_Y:
#         indicator_sum = sum(kernel_weights[i] * (Y[i] <= q) for i in range(n))
#         if indicator_sum / sum(kernel_weights) >= quantile:
#             return q
        
#     # TODO: 使用fixed的精度
#     # 二分方法
#     return None

import numpy as np

def predict_kernel_quantile(X, Y, X_new, quantile, epsilon=1e-5):
    n = len(X)
    kernel_weights = np.array([kernel_function(X_new, X[i]) for i in range(n)])
    print(n)
    low, high = min(Y), max(Y)
    
    while high - low > epsilon:
        mid = (high + low) / 2
        indicator_sum = sum(kernel_weights[i] * (Y[i] <= mid) for i in range(n))
        
        if indicator_sum / sum(kernel_weights) < quantile:
            low = mid
        else:
            high = mid
    
    return (high + low) / 2



def perform_regression_analysis(X, Y, train_ratio, test_ratio, validation_ratio, quantile, model_type):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1 - train_ratio, random_state=0)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=validation_ratio/(test_ratio + validation_ratio), random_state=0)
    
    n_features = X_train.shape[1]
    if model_type in ['linear', 'lasso', 'ridge', 'quantile', 'glm']:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        X_validation = sm.add_constant(X_validation)

    if model_type == 'linear':
        model = LinearRegression().fit(X_train, Y_train)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1).fit(X_train, Y_train)  
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0).fit(X_train, Y_train)  
    elif model_type == 'quantile':
        model = sm.QuantReg(Y_train, X_train).fit(q=quantile)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=0).fit(X_train, Y_train)
    elif model_type == 'glm':
        model = sm.GLM(Y_train, X_train, family=sm.families.Gaussian()).fit()
    elif model_type == 'neural_network':
        nn_model = train_neural_network(X_train, Y_train)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_validation = torch.tensor(X_validation, dtype=torch.float32)
        Y_pred_test = nn_model(X_test).detach().numpy().flatten()
        Y_pred_validation = nn_model(X_validation).detach().numpy().flatten()
    elif model_type == 'ko':
        print("1")
        Y_pred_test = np.array([predict_kernel_quantile(X, Y, x, quantile) for x in X_test])
        Y_pred_validation = np.array([predict_kernel_quantile(X, Y, x, quantile) for x in X_validation])
    elif model_type == 'quantile_net':
        model = train_quantile_network(X_train, Y_train, quantile, n_features)
        Y_pred_test = model.predict(X_test)
        Y_pred_validation = model.predict(X_validation)
    else:
        raise ValueError("Invalid model type provided.")

    # Predicting and calculating errors
    if model_type not in ['neural_network', 'ko', 'quantile_net']:
        Y_pred_test = model.predict(X_test)
        Y_pred_validation = model.predict(X_validation)

    E_i = Y_test - Y_pred_test

    adjusted_quantile = quantile * (1 + 1 / len(E_i))
    Q_alpha_E = np.quantile(E_i, adjusted_quantile)
    Y_pred_validation_adjusted = Y_pred_validation + Q_alpha_E

    # Loss calculation

    loss_unadjusted = quantile_loss(Y_validation, Y_pred_test, quantile)
    loss_adjusted = quantile_loss(Y_validation, Y_pred_validation_adjusted, quantile)
    loss = (loss_unadjusted, loss_adjusted)

    print(model_type, "loss unadjusted", loss_unadjusted, "loss_adjusted", loss_adjusted)
    return loss

def train_neural_network(X_train, Y_train):
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

    # Neural network
    model = SimpleNN(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 500

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

    return model
def train_quantile_network(X_train, Y_train, quantile, n_features):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

    # Neural network
    model = tf.keras.Sequential([
        layers.Dense(30, activation='relu', input_shape=(n_features,)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))
    model.fit(X_train, Y_train, epochs = 100, batch_size=32, verbose=1)

    return model

def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()
