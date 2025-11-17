# test_pipeline.py
import ee
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Step 1: Initialize Google Earth Engine
# -----------------------------
# Authenticate your account (this opens a browser window on first run)
ee.Authenticate()  

# Initialize without a placeholder project
ee.Initialize(project='digitaltwin-478518')  
print("Google Earth Engine initialized successfully!")

point = ee.Geometry.Point([-0.1276, 51.5074])
print("EE Point info:", point.getInfo())

# -----------------------------
# Step 2: Test TensorFlow/Keras setup
# -----------------------------
# Create dummy data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.rand(100, 1)  # 100 target values

# Build a tiny ANN
model = Sequential([
    Dense(8, input_shape=(5,), activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Train for a few epochs
model.fit(X, y, epochs=3, batch_size=16)
print("Dummy ANN ran successfully!")

# -----------------------------
# Step 3: Test Earth Engine (optional)
# -----------------------------
# Example: get the coordinates of a location
point = ee.Geometry.Point([-0.1276, 51.5074])  # London coordinates
print("Example EE object:", point.getInfo())
