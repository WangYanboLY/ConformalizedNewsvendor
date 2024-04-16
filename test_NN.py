import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Generate 15-dimensional data X with normal distribution
n_samples = 1000
n_features = 15
X = np.random.normal(0, 1, (n_samples, n_features))

# Step 2: Create Y with a linear relationship to X
# Random coefficients for linear relationship and some noise
coefficients = np.random.normal(1, 4, n_features)
Y = np.dot(X, coefficients) + np.random.normal(0, 0.5, n_samples)

# Step 3: Split the data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verify the shape of the datasets
(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# Neural Network Model Creation
model = Sequential()
model.add(Dense(64, input_dim=n_features, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the Model
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the Model
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Creating the models
models = {
    "Regression Tree": DecisionTreeRegressor(),
    "SVR": SVR(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest Regression": RandomForestRegressor()
}

# Training and Evaluating each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, Y_train)

    # Evaluate the model
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"{name} - Test MSE: {mse}")

