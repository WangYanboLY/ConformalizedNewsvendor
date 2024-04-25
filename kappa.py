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
    low, high = min(Y), max(Y)
    
    while high - low > epsilon:
        mid = (high + low) / 2
        indicator_sum = sum(kernel_weights[i] * (Y[i] <= mid) for i in range(n))
        if indicator_sum / sum(kernel_weights) < quantile:
            low = mid
        else:
            high = mid
    
    return (high + low) / 2



def calculate_kappa_in(X, Y, quantile, model_type, quantile_real):
    n_features = X.shape[1]
    n = X.shape[0]
    if model_type in ['linear', 'lasso', 'ridge', 'quantile', 'glm']:
        X = sm.add_constant(X)

    if model_type == 'linear':
        model = LinearRegression().fit(X, Y)  
    elif model_type == 'quantile':
        model = sm.QuantReg(Y, X).fit(q=quantile)
    elif model_type == 'ko':
        Y_pred = np.array([predict_kernel_quantile(X, Y, x, quantile) for x in X])
    elif model_type == 'quantile_net':
        model = train_quantile_network(X, Y, quantile, n_features)
        X = torch.tensor(X, dtype=torch.float32)
        Y_pred = model(X).detach().numpy().flatten()
    else:
        raise ValueError("Invalid model type provided.")

    # Predicting and calculating errors
    if model_type not in ['neural_network', 'ko', 'quantile_net']:
        Y_pred = model.predict(X)

    bias = []
    kappa = []
    d = []
    for i in range(n):
        bias.append(Y_pred[i] - quantile_real[i])

    for i in range(n):
        for j in range(n):
            kappa.append(np.abs(bias[i] - bias[j]))
            d.append(np.linalg.norm(X[i] - X[j]))
    return kappa, d, bias, Y_pred

def calculate_kappa_ext(X, Y, quantile, ratio, model_type, quantile_real):
    n_features = X.shape[1]

    X_train, X_test, Y_train, Y_test, quantile_train, quantile_test= train_test_split(X, Y, quantile_real, test_size=1 - ratio,
                                                        random_state=0)
    if model_type in ['linear', 'lasso', 'ridge', 'quantile', 'glm']:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

    if model_type == 'linear':
        model = LinearRegression().fit(X_train, Y_train)  
    elif model_type == 'quantile':
        model = sm.QuantReg(Y_train, X_train).fit(q=quantile)
    elif model_type == 'ko':
        Y_pred = np.array([predict_kernel_quantile(X_train, Y_train, x, quantile) for x in X_test])
    elif model_type == 'quantile_net':
        model = train_quantile_network(X_train, Y_train, quantile, n_features)
        X_test= torch.tensor(X_test, dtype=torch.float32)
        Y_pred = model(X_test).detach().numpy().flatten()
    else:
        raise ValueError("Invalid model type provided.")

    # Predicting and calculating errors
    if model_type not in ['neural_network', 'ko', 'quantile_net']:
        Y_pred = model.predict(X_test)
    n = quantile_test.shape[0]
    bias = []
    kappa = []
    d = []
    for i in range(n):
        bias.append(Y_pred[i] - quantile_test[i])

    for i in range(n):
        for j in range(n):
            kappa.append(np.abs(bias[i] - bias[j]))
            d.append(np.linalg.norm(X_test[i] - X_test[j]))
    return kappa, d, bias

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

    model = nn.Sequential(
        nn.Linear(n_features, 30),
        nn.ReLU(),
        nn.Linear(30, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = quantile_loss_neural_net(quantile, Y_train, outputs)
        loss.backward()
        optimizer.step()

    return model
def quantile_loss_neural_net(quantile, Y_train, outputs):
    errors = Y_train - outputs
    return torch.max((quantile - 1) * errors, quantile * errors).mean()


def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()




def perform_jackknife_plus(X, Y, validation_ratio, quantile, model_type):
# Splitting the dataset into training and validation sets
    if model_type in ['linear', 'lasso', 'ridge', 'quantile', 'glm']:
        X = sm.add_constant(X)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_ratio, random_state=0)

    # Initialize lists to store predictions and adjustments
    predictions = []
    adjustments = []

    # Loop over all data points in the training set
    for i in range(len(X_train)):
        # Create a training set excluding the current data point
        X_train_i = np.delete(X_train, i, axis=0)
        Y_train_i = np.delete(Y_train, i, axis=0)

        # Train the model on this modified training set
        model_i = train_model(X_train_i, Y_train_i, model_type, quantile)

        # Predict for the left-out data point and for the new data point in validation set
        Y_pred_left_out = model_i.predict(X_train[i].reshape(1, -1))
        Y_pred_validation = model_i.predict(X_validation)

        # Calculate the adjustment for the left-out data point
        adjustment = Y_train[i] - Y_pred_left_out

        # Store predictions and adjustments
        predictions.append(Y_pred_validation)
        adjustments.append(adjustment)

    # Compute final adjusted predictions
    predictions = np.array(predictions)
    adjustments = np.array(adjustments)
    adjusted_predictions = np.quantile(predictions + adjustments, quantile, axis=0)

    # Compute loss on validation set
    loss = quantile_loss(Y_validation, adjusted_predictions, quantile)

    print(model_type, "jackknife+ loss", loss)
    return loss

def train_model(X, Y, model_type, quantile):
    # Preprocessing for models that require adding a constant

    # Training the model based on the specified type
    if model_type == 'linear':
        model = LinearRegression().fit(X, Y)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1).fit(X, Y)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0).fit(X, Y)
    elif model_type == 'quantile':
        model = sm.QuantReg(Y, X).fit(q=quantile)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=0).fit(X, Y)
    elif model_type == 'glm':
        model = sm.GLM(Y, X, family=sm.families.Gaussian()).fit()
    elif model_type == 'neural_network':
        # Assuming `train_neural_network` is a function defined elsewhere to train a neural network
        model = train_neural_network(X, Y)
    else:
        raise ValueError("Invalid model type provided.")
    
    return model
