import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache
def load_data():
    try:
        data = pd.read_excel("syncope_heat_conditions_dataset_1000_patients.xlsx")
        return data
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure the dataset file is uploaded correctly.")
        return None

# Function to preprocess data
def preprocess_data(data):
    if data is None:
        return None, None
    # Convert categorical data (like Tilt Table Test) to numerical
    data['Tilt Table Test (Positive/Negative)'] = data['Tilt Table Test (Positive/Negative)'].map({"Positive": 1, "Negative": 0})

    # Map diagnosis to numerical values for classification
    diagnosis_map = {
        "Vasovagal Syncope": 0,
        "Cardiogenic Syncope": 1,
        "Orthostatic Syncope": 2,
        "Heat Exhaustion": 3,
        "Heat Stroke": 4
    }
    data['Diagnosis'] = data['Diagnosis (Jenis Sinkop atau Heat Stroke/Exhaustion)'].map(diagnosis_map)

    features = data.drop(columns=['ID Pasien', 'Gejala (Kalimat Paragraf)', 'Diagnosis (Jenis Sinkop atau Heat Stroke/Exhaustion)'])
    target = data['Diagnosis']

    return features, target

# Load and preprocess dataset
data = load_data()
features, target = preprocess_data(data)

# Check if data is loaded and preprocessed
if features is not None and target is not None:
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model accuracy on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

# Streamlit app interface
st.title("AI Diagnosis for Syncope and Heat-Related Conditions")

# Input from user: a paragraph describing the case
input_text = st.text_area("Enter the patient's case description:", "Patient feels dizzy after standing in a hot environment for a long time.")

if input_text:
    if features is not None:
        # Example: Using static input features for demonstration
        blood_pressure = 90  # Static for demonstration
        heart_rate = 100
        cardiac_output = 5.0
        lactate_level = 2.5
        lpr = 12.0
        agma = 16.0
        body_temp = 39.5
        map_bp = 80.0
        svr = 1200

        # Prepare input features
        input_features = np.array([[blood_pressure, heart_rate, cardiac_output, lactate_level, lpr, agma, 
                                    body_temp, 1 if "faint" in input_text else 0, 20, map_bp, svr]])

        # Predict diagnosis
        prediction = model.predict(input_features)[0]
        confidence = model.predict_proba(input_features).max()

        # Map prediction back to diagnosis
        diagnosis_map_reverse = {
            0: "Vasovagal Syncope",
            1: "Cardiogenic Syncope",
            2: "Orthostatic Syncope",
            3: "Heat Exhaustion",
            4: "Heat Stroke"
        }

        # Threshold for confidence, below which no diagnosis is given
        confidence_threshold = 0.7

        if confidence >= confidence_threshold:
            diagnosis_result = diagnosis_map_reverse[prediction]
            # Display diagnosis
            st.subheader("Predicted Diagnosis")
            st.write(f"Diagnosis: **{diagnosis_result}**")
            st.write(f"Model Confidence: **{confidence * 100:.2f}%**")
        else:
            # If confidence is too low, provide feedback
            st.subheader("Predicted Diagnosis")
            st.write(f"Diagnosis: **Could not determine a specific condition**")
            st.write(f"Model Confidence: **{confidence * 100:.2f}% (too low for a conclusive diagnosis)**")

        # Display model performance
        st.subheader("Model Performance")
        st.write(f"Model Accuracy on Test Set: **{accuracy * 100:.2f}%**")

        # Display calculated clinical parameters
        st.subheader("Calculated Clinical Parameters")
        st.write(f"Blood Pressure: {blood_pressure:.2f} mmHg")
        st.write(f"Heart Rate: {heart_rate:.2f} bpm")
        st.write(f"Lactate to Pyruvate Ratio (LPR): {lpr:.2f}")
        st.write(f"Anion Gap Metabolic Acidosis (AGMA): {agma:.2f}")
        st.write(f"Cardiac Output (CO): {cardiac_output:.2f} L/min")
        st.write(f"Mean Arterial Pressure (MAP): {map_bp:.2f} mmHg")
        st.write(f"Systemic Vascular Resistance (SVR): {svr:.2f} dynes·sec·cm⁻⁵")
        st.write(f"Body Temperature: {body_temp:.2f} °C")
    else:
        st.error("Model could not be initialized due to missing dataset.")
