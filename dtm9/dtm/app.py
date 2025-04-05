from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

app = Flask(__name__)

# Load data from the Excel file (ensure this matches your training data path)
file_path = "Extended_Cricket_Shot_Dataset.xlsx"  # Update with your file path
data = pd.read_excel(file_path)

heights = data["height"].values
weights = data["weight"].values
bmi = data["bmi"].values
ages = data["age"].values
ratings = data.iloc[:, 4:].values  # Assuming ratings start from the fifth column onward
shot_names = data.columns[4:]

# Define the model architecture (should match training architecture)
def create_model(n_shots):
    height_input = Input(shape=(1,), name="height")
    weight_input = Input(shape=(1,), name="weight")
    bmi_input = Input(shape=(1,), name="bmi")
    age_input = Input(shape=(1,), name="age")

    person_features = Concatenate()([height_input, weight_input, bmi_input, age_input])
    
    x = Dense(128, activation="relu")(person_features)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)

    output = Dense(n_shots, activation="linear")(x)

    model = Model(inputs=[height_input, weight_input, bmi_input, age_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model

# Create and load the model weights for inference
n_shots = ratings.shape[1]
model = create_model(n_shots)

try:
    # Attempt to load weights with options for name matching and skipping mismatches if necessary.
    model.load_weights('model_weights.h5', by_name=True)  # or use skip_mismatch=True if necessary.
except Exception as e:
    print(f"Error loading model weights: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            age = float(request.form['age'])

            # Calculate BMI
            bmi_value = weight / ((height / 100) ** 2)

            # Normalize inputs using training data normalization values.
            norm_height = (height - np.mean(heights)) / (np.std(heights) + 1e-7)
            norm_weight = (weight - np.mean(weights)) / (np.std(weights) + 1e-7)
            norm_bmi_value = (bmi_value - np.mean(bmi)) / (np.std(bmi) + 1e-7)
            norm_age = (age - np.mean(ages)) / (np.std(ages) + 1e-7)

            new_user_inputs = np.array([[norm_height], [norm_weight], [norm_bmi_value], [norm_age]])
            
            # Make prediction using normalized inputs.
            predicted_ratings = model.predict([new_user_inputs[0], new_user_inputs[1], new_user_inputs[2], new_user_inputs[3]])

            results_dict = {shot_names[i]: predicted_ratings[0][i] for i in range(len(shot_names))}
        except Exception as e:
            results_dict = {"error": str(e)}
        
        return render_template('index.html', results=results_dict)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)