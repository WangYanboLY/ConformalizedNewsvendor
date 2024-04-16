import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def quantile_loss(y_true, y_pred, quantile):
    e = y_true - y_pred
    loss = torch.max(quantile * e, (quantile - 1) * e)
    return torch.mean(loss)


class QuantileNet(nn.Module):
    def __init__(self, input_size):
        super(QuantileNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, X_train, Y_train, quantile, epochs=100, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = lambda y_true, y_pred: quantile_loss(y_true, y_pred, torch.tensor(quantile))

    dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(targets, outputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

def predict(model, X):
    model.eval()  
    with torch.no_grad():
        X_tensor = torch.Tensor(X)
        predictions = model(X_tensor)
    return predictions.numpy()


class L2Net(nn.Module):
    def __init__(self, input_size):
        super(L2Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)  
        self.fc2 = nn.Linear(10, 8)              
        self.fc3 = nn.Linear(8, 1)           
        self.relu = nn.ReLU()                  

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)            
        return x
def train_model_L2(model, X_train, Y_train, epochs=100, batch_size=32, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

quantile = 0.9
n_samples = 100000
n_features = 30
np.random.seed(0)

X = np.random.normal(0, 100, (n_samples, n_features))
coefficients = np.random.normal(10, 400, n_features)
noise = np.random.normal(0, 16, n_samples)

Y = np.dot(X, coefficients) + noise

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1 - train_ratio, random_state=0)
X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=validation_ratio/(test_ratio + validation_ratio), random_state=0)

model = QuantileNet(input_size=X_train.shape[1])
ANN_model = train_model(model, X_train, Y_train, quantile)

modelL2 = L2Net(input_size=X_train.shape[1])
model_L2_trained = train_model_L2(modelL2, X_train, Y_train)

# quantile loss net
Y_pred_test_ANN = predict(ANN_model, X_test)
E_ANN = Y_test - Y_pred_test_ANN.flatten()
adjusted_quantile_ANN = quantile * (1 + 1 / len(E_ANN))
Q_ANN = np.quantile(E_ANN, adjusted_quantile_ANN)
Y_pred_validation_ANN = predict(ANN_model, X_validation)
Y_pred_validation_ANN_adjusted = Y_pred_validation_ANN.flatten() + Q_ANN

# l2 loss model
Y_pred_test_ANNL2 = predict(model_L2_trained, X_test)
E_ANN2 = Y_test - Y_pred_test_ANNL2.flatten()
adjusted_quantile_ANN2 = quantile * (1 + 1 / len(E_ANN2))
Q_ANN2 = np.quantile(E_ANN2, adjusted_quantile_ANN2)
Y_pred_validation_ANN2 = predict(model_L2_trained, X_validation)
Y_pred_validation_ANN_adjusted2 = Y_pred_validation_ANN2.flatten() + Q_ANN2

def quantile_loss_1(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()
quantile_loss_unadjusted_ANN = quantile_loss_1(Y_validation, Y_pred_validation_ANN, quantile)
quantile_loss_adjusted_ANN = quantile_loss_1(Y_validation, Y_pred_validation_ANN_adjusted, quantile)

quantile_loss_unadjusted_ANN2 = quantile_loss_1(Y_validation, Y_pred_validation_ANN2, quantile)
quantile_loss_adjusted_ANN2 = quantile_loss_1(Y_validation, Y_pred_validation_ANN_adjusted2, quantile)

print("ANN loss:", quantile_loss_unadjusted_ANN, "\n",  "ANN loss adjusted:", quantile_loss_adjusted_ANN)

print("ANN loss L2:", quantile_loss_unadjusted_ANN2, "\n",  "ANN loss adjusted L@:", quantile_loss_adjusted_ANN2)
