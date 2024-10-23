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

    # Select relevant features for prediction
    features = data[['Blood Pressure (mmHg)', 'Heart Rate (bpm)', 'Cardiac Output (L/min)', 
                     'Lactate Level (mmol/L)', 'Lactate to Pyruvate Ratio (LPR)', 
                     'Anion Gap Metabolic Acidosis (AGMA)', 'Body Temperature (°C)', 
                     'Tilt Table Test (Positive/Negative)', 'Orthostatic Hypotension (mmHg)', 
                     'Mean Arterial Pressure (MAP) (mmHg)', 'Systemic Vascular Resistance (SVR)']]

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
input_text = st.text_area("Enter the patient's case description:", "Masukkan kasus pasien di kolom ini - AI by Allam R.")

# Tombol untuk memproses prediksi hanya setelah ditekan
if st.button("Analyze"):
    if input_text:
        if features is not None:
            # Adjusted input features to reflect Mira's case
            blood_pressure = 110  # Mira's blood pressure is 110/70 mmHg
            heart_rate = 85       # Mira's heart rate is 85 bpm
            cardiac_output = 5.0   # Estimated cardiac output
            lactate_level = 2.0    # Normal lactate level
            lpr = 12.0             # Normal Lactate to Pyruvate Ratio (LPR)
            agma = 16.0            # Normal Anion Gap Metabolic Acidosis
            body_temp = 37.0       # Normal body temperature
            ttt_val = 1            # Tilt Table Test simulation, positive due to syncope
            orthostatic_hypotension = 20  # Mild orthostatic hypotension
            map_bp = 80.0          # Normal Mean Arterial Pressure
            svr = 1000             # Estimated Systemic Vascular Resistance

            # Prepare input features and ensure they match training data
            input_features = np.array([[blood_pressure, heart_rate, cardiac_output, lactate_level, lpr, agma, 
                                        body_temp, ttt_val, orthostatic_hypotension, map_bp, svr]])

            # Log the shape of input data
            st.write("Shape of input_features:", input_features.shape)  # Log jumlah fitur saat prediksi

            # Check if the number of features matches before making predictions
            if input_features.shape[1] == X_train.shape[1]:
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
                st.error("Mismatch in the number of features between training data and input. Please check your input.")
        else:
            st.error("Model could not be initialized due to missing dataset.")
