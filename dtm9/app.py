# Import necessary libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (assuming model is saved as 'cricket_shot_model.h5')
model = tf.keras.models.load_model('cricket_shot_model.tf')

# Load data for normalization and shot information (assuming the same file path)
file_path = "Extended_Cricket_Shot_Dataset.xlsx"
data = pd.read_excel(file_path)

# Separate necessary data for normalization and mapping
heights = data["height"].values
weights = data["weight"].values
bmi = data["bmi"].values
ages = data["age"].values
shot_names = data.columns[4:]
ratings = data.iloc[:, 4:].values
epsilon = 1e-7

# Define shot info dictionary (as in your original code)
shot_info = {  "Straight Drive": {"link": "https://youtu.be/8eb68_qqFHM?si=43OfKgsEMtgYqcjD",
                       "guidance": "Keep a stable stance and let the ball come to you. Lean forward and drive it with the full face of the bat."},
    "Pull Shot": {"link": "https://youtu.be/g_Y_y4OHm1o?si=ejAO8C5cQW3Hb3Zz",
                  "guidance": "Swivel on the back foot, keep the bat horizontal, and use timing to send the ball to the leg side."},
    "Lofted Shot": {"link": "https://youtu.be/QuiTxf_G-8A?si=BQL0Ic1uJwLhVWQ0",
                    "guidance": "Position your feet well, lean slightly back, and use your bottom hand to generate lift."},
    "Cover Drive": {"link": "https://youtu.be/JgGf_5LY6qA?si=1sRvstpzNXd6s15-",
                    "guidance": "Step forward with the front foot, align with the ball, and drive with a flowing bat movement."},
    "Hook Shot": {"link": "https://youtu.be/quvpwDBViFg?si=v-74KSF3Zx0ySkK_",
                  "guidance": "Play on the back foot with good hand-eye coordination, aiming to hook short balls on the leg side."},
    "Sweep Shot": {"link": "https://youtu.be/VAuatgBXwDA?si=ZGOsg3EbqtfZf4TL",
                   "guidance": "Stay low, watch the ball onto the bat, and sweep it to the leg side by rolling your wrists."},
    "Leg Glance": {"link": "https://youtu.be/7--nHWEqt_k?si=lYQvuwY4hzz6MMTh",
                   "guidance": "Turn the bat face slightly and use soft hands to direct the ball toward fine leg."},
    "Square Cut": {"link": "https://youtu.be/p4WPqPirEes?si=yNn33ixRyGdh61Mx",
                   "guidance": "Position yourself on the back foot, use timing to cut the ball square to the off-side."},
    "On Drive": {"link": "https://youtu.be/KLfmgX-0LtI?si=RG0VrkQkOjGFqvFK",
                 "guidance": "Step forward with a slight inward angle, and drive the ball straight down the ground on the leg side."},
    "Off Drive": {"link": "https://youtu.be/kAtrMpWiClw?si=VIcdldNPEnBtEbU6",
                  "guidance": "Position yourself with a straight stance and use the full face of the bat to drive the ball past the bowler."},
    "Reverse Sweep": {"link": "https://youtu.be/WEvPkB1Za8s?si=i-CTwlLjaMt9NafH",
                      "guidance": "Quickly adjust your hands and feet to sweep the ball to the off side, ideal for shorter deliveries."},
    "Scoop Shot": {"link": "https://youtu.be/sYkw-US0tEU?si=_ud0uANORlNMzrjG",
                   "guidance": "Get into position early, use the bottom hand to lift the ball over the keeper or fine leg."},
    "Upper Cut": {"link": "https://youtu.be/jWLdZDBFZao?si=PF-j_HVEaqLZJ12d",
                  "guidance": "Play with an angled bat to glide short balls over the slips or point, suitable for bouncers."},
    "Flick Shot": {"link": "https://youtu.be/scWEE1ERNl8?si=r6C2IrSGwl5GS8KJ",
                   "guidance": "Rotate your wrists and use the bottom hand to flick balls on the leg side, keeping the head still."},
    "Paddle Sweep": {"link": "https://youtu.be/PQJgP2IbuvE?si=9wRaaf1GNAivd_Z-",
                     "guidance": "Stay low, use soft hands to deflect the ball fine on the leg side. Requires good wrist control."} }  # Use your full shot_info dictionary here

# Define weekly schedule
weekly_schedule = {  "Monday": "Practice basics of grip, stance, and footwork. Cardio workout (30 mins). High-protein meals.",
    "Tuesday": "Batting drills: Focus on front foot shots. Strength training. Balanced diet with healthy carbs.",
    "Wednesday": "Practice back foot shots. Core workouts. Light meals with vegetables and lean protein.",
    "Thursday": "Running drills for agility. Batting session to review shots. Protein-rich foods and hydration.",
    "Friday": "Practice underarm and throw-down drills. Strength training. Nutritious snacks and balanced dinner.",
    "Saturday": "Mock matches to test skills in game-like situations. Rest after practice. Energy-focused meals.",
    "Sunday": "Rest day with light stretching. Hydrate well and focus on balanced nutrition." }  # Use your full weekly_schedule dictionary here

# Define normalization function
def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + epsilon)

# Web route for home page with form
@app.route('/')
def index():
    return render_template('index.html')

# Web route to handle the form submission
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Retrieve user inputs
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        age = float(request.form['age'])
        bmi = weight / ((height / 100) ** 2)
        
        # Normalize inputs
        norm_height = (height - np.mean(heights)) / (np.std(heights) + epsilon)
        norm_weight = (weight - np.mean(weights)) / (np.std(weights) + epsilon)
        norm_bmi = (bmi - np.mean(bmi)) / (np.std(bmi) + epsilon)
        norm_age = (age - np.mean(ages)) / (np.std(ages) + epsilon)

        # Predict ratings
        predicted_ratings = model.predict([[norm_height, norm_weight, norm_bmi, norm_age]])[0]
        
        # Prepare results
        shot_recommendations = sorted(
            [(shot_names[i], predicted_ratings[i], shot_info.get(shot_names[i])) 
             for i in range(len(shot_names))],
            key=lambda x: x[1], reverse=True
        )
        
        # Render the results
        return render_template(
            'results.html', 
            shot_recommendations=shot_recommendations[:5], 
            weekly_schedule=weekly_schedule
        )
    except Exception as e:
        return str(e)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
