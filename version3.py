import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Define the quantile loss function
def quantile_loss(q, y_true, y_pred):
    e = (y_true - y_pred)
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), axis=-1)

# Number of samples and dimension of X
num_samples = 1000
dim_x = 15

# Generate random data for X
X = np.random.normal(10, 100, (num_samples, dim_x))

# Generate coefficients for the linear relationship
coefficients = np.random.normal(1.5, 16, dim_x)

# Generate Y with a linear relationship with X
Y = np.dot(X, coefficients) + np.random.normal(0, 0.1, num_samples)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential([
    Dense(20, input_dim=dim_x, activation='relu'),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1)
])

# Compile the model with quantile loss
quantile = 0.9
model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))

# Train the model
model.fit(X_train, Y_train, epochs=160, batch_size=32, verbose=1)


model1 = Sequential([
    Dense(64, input_dim=dim_x, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model1.compile(optimizer='adam', loss='mean_squared_error')

model1.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, Y_test, verbose=0)
test_loss1 = model1.evaluate(X_test, Y_test, verbose = 0)
print(f"Average Quantile Loss on Test Set: {test_loss}")
print(f"Average Quantile Loss on Test Set with mse: {test_loss1}")
