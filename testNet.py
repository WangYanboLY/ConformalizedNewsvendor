import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import statsmodels.api as sm

np.random.seed(0)  
n_samples = 10000
n_features = 30
# X = np.random.randn(n_samples, n_features)
# beta = np.random.randn(n_features, 1)
# epsilon = np.random.randn(n_samples, 1)
# Y = np.dot(X, beta) + epsilon

X = np.random.normal(0, 100, (n_samples, n_features))
coefficients = np.random.normal(1, 10, n_features)
noise = np.random.normal(0, 16, n_samples)
Y = np.dot(X, coefficients) + noise

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1 - train_ratio, random_state=0)
X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=validation_ratio/(test_ratio + validation_ratio), random_state=0)

from sklearn.model_selection import train_test_split
def quantile_loss(q, y_true, y_pred):
    e = (y_true - y_pred)
    return tf.keras.backend.mean(tf.keras.backend.maximum(q * e, (q - 1) * e), axis=-1)
quantile = 0.9  



## Quantile Neural Network
model = tf.keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=(n_features,)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))
model.fit(X_train, Y_train, epochs = 100, batch_size=32, verbose=1)
y_pred = model.predict(X_test)
E_i = Y_test-y_pred
adjusted_quantile = quantile * (1 + 1 / len(E_i))
Q_alpha_E = np.quantile(E_i, adjusted_quantile)
Y_pred_validation = model.predict(X_validation)
Y_pred_validation_adjusted = Y_pred_validation + Q_alpha_E
quantile_loss_unadjusted_QNN = np.mean(quantile_loss(quantile, Y_validation, Y_pred_validation))
quantile_loss_adjusted_QNN = np.mean(quantile_loss(quantile, Y_validation, Y_pred_validation_adjusted))

## L2 Neural Network
model1 = tf.keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=(n_features,)),
    layers.Dense(1)
])
model1.compile(optimizer='adam', loss='mean_squared_error')
model1.fit(X_train, Y_train, epochs = 100, batch_size=32, verbose=1)
y_pred_1 = model1.predict(X_test)
E_i_1 = Y_test - y_pred_1
adjusted_quantile_1 = quantile * (1 + 1/len(E_i_1))
Q_alpha_E_1 = np.quantile(E_i_1, adjusted_quantile_1)
Y_1_pred_validation = model1.predict(X_validation)
Y_1_pred_validation_adjusted = Y_1_pred_validation + Q_alpha_E_1
quantile_loss_unadjusted_1 = np.mean(quantile_loss(quantile, Y_validation, Y_1_pred_validation))
quantile_loss_adjusted_1 = np.mean(quantile_loss(quantile, Y_validation, Y_1_pred_validation_adjusted))

## Linear model
def quantile_loss_np(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error).mean()
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
quantile_reg_model = sm.QuantReg(Y_train, X_train_const).fit(q=quantile)
linear_model = sm.OLS(Y_train, X_train_const).fit()
Y_pred_test = quantile_reg_model.predict(X_test_const)
Y_linear_pred_test = linear_model.predict(X_test_const)

E_i = Y_test - Y_pred_test
E_i_linear = Y_test - Y_linear_pred_test

adjusted_quantile = quantile * (1 + 1 / len(E_i))
adjusted_quantile_linear = quantile*(1 + 1 / len(E_i_linear))

Q_alpha_E = np.quantile(E_i, adjusted_quantile)
Q_alpha_E_linear = np.quantile(E_i_linear, adjusted_quantile_linear)
def calculate_quantile(Y_train, Y_validation, alpha):
    Y_pred_online_SAA= []
    for i in range(len(Y_validation)):
        combined_data = np.concatenate([Y_train, Y_validation[:i]])
        quantile = np.quantile(combined_data, alpha)
        Y_pred_online_SAA.append(quantile)
    return Y_pred_online_SAA
Y_pred_test_SAA = calculate_quantile(Y_train, Y_test, quantile)
E_SAA = Y_test - Y_pred_test_SAA
adjusted_quantile_SAA = quantile * (1 + 1 / len(E_SAA))
Q_SAA = np.quantile(E_SAA, adjusted_quantile_SAA)

X_validation_const = sm.add_constant(X_validation)

Y_pred_validation = quantile_reg_model.predict(X_validation_const)
Y_pred_validation_adjusted = Y_pred_validation + Q_alpha_E
Y_pred_validation_linear = linear_model.predict(X_validation_const)
Y_pred_validation_linear_adjusted = Y_pred_validation_linear + Q_alpha_E_linear

Y_pred_SAA = np.quantile(Y_train, quantile)
Y_pred_validation_online_SAA = calculate_quantile(Y_train, Y_validation, quantile)
Y_SAA_adjusted = Y_pred_validation_online_SAA + Q_SAA

quantile_loss_unadjusted = quantile_loss_np(Y_validation, Y_pred_validation, quantile)
quantile_loss_adjusted = quantile_loss_np(Y_validation, Y_pred_validation_adjusted, quantile)
quantile_loss_linear = quantile_loss_np(Y_validation, Y_pred_validation_linear, quantile)
quantile_loss_linear_adjusted = quantile_loss_np(Y_validation, Y_pred_validation_linear_adjusted, quantile)
quantile_loss_SAA = quantile_loss_np(Y_validation, Y_pred_SAA, quantile)
quantile_loss_online_SAA = quantile_loss_np(Y_validation, Y_pred_validation_online_SAA, quantile)
quantile_loss_SAA_adjusted = quantile_loss_np(Y_validation, Y_SAA_adjusted, quantile)

print("Quantile Loss Unadjusted:", quantile_loss_unadjusted, "\n",  "Quantile Loss Adjusted:", quantile_loss_adjusted, "\n", 
      "Linear loss: ", quantile_loss_linear,  "\n","Linear adjusted loss:", quantile_loss_linear_adjusted, "\n", 
      "SAA quantile loss: ", quantile_loss_SAA, "\n", "online SAA quantile loss:", quantile_loss_online_SAA, "\n", 
      "SAA adjusted:", quantile_loss_SAA_adjusted)

print("QNN loss:", quantile_loss_unadjusted, "\n",  "QNN loss adjusted:", quantile_loss_adjusted)
print("L2NN loss:", quantile_loss_unadjusted_1, "\n",  "L2NN loss adjusted:", quantile_loss_adjusted_1)