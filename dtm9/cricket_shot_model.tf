# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data from the Excel file
file_path = "Extended_Cricket_Shot_Dataset.xlsx"
data = pd.read_excel(file_path)

# Separate features and ratings
heights = data["height"].values
weights = data["weight"].values
bmi = data["bmi"].values
ages = data["age"].values
ratings = data.iloc[:, 4:].values  # Assuming ratings start from the fifth column onward
shot_names = data.columns[4:]  # Get shot names for recommendation mapping

# Define the number of cricket shots
n_shots = ratings.shape[1]

# Split data into training and testing sets
train_heights, test_heights, train_weights, test_weights, train_bmi, test_bmi, train_ages, test_ages, train_ratings, test_ratings = train_test_split(
    heights, weights, bmi, ages, ratings, test_size=0.2, random_state=42
)

# Apply normalization with a small epsilon to avoid division by zero
epsilon = 1e-7
def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + epsilon)

train_heights = normalize(train_heights)
test_heights = normalize(test_heights)
train_weights = normalize(train_weights)
test_weights = normalize(test_weights)
train_bmi = normalize(train_bmi)
test_bmi = normalize(test_bmi)
train_ages = normalize(train_ages)
test_ages = normalize(test_ages)

# Define the model
def create_complex_model(n_shots):
    # Input layers
    height_input = Input(shape=(1,), name="height")
    weight_input = Input(shape=(1,), name="weight")
    bmi_input = Input(shape=(1,), name="bmi")
    age_input = Input(shape=(1,), name="age")

    # Concatenate features
    person_features = Concatenate()([height_input, weight_input, bmi_input, age_input])

    # Dense layers with BatchNormalization and Dropout
    x = Dense(128, activation="relu")(person_features)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)

    # Output layer for predicted ratings
    output = Dense(n_shots, activation="linear")(x)

    # Compile the model
    model = Model(inputs=[height_input, weight_input, bmi_input, age_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Instantiate and train the model
model = create_complex_model(n_shots)
history = model.fit(
    [train_heights, train_weights, train_bmi, train_ages],
    train_ratings,
    validation_data=([test_heights, test_weights, test_bmi, test_ages], test_ratings),
    epochs=100,  # Set epochs to 100
    batch_size=32,
    verbose=1
)

# Save the trained model as 'cricket_shot_model.h5'
model.save('cricket_shot_model.h5')

# Optional: Plot training history
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# MAE Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
