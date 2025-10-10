# ===============================================================
# 🌤️ STREAMLIT WEATHER FORECASTING APP
# Uses Multi-Output LSTM to Predict Temp, Humidity, and Pressure
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import warnings

# Suppress Keras warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- Global Constants ---
SEQ_LEN = 24  # Look-back window in hours
FEATURES = [
    "Temperature (C)", 
    "Humidity", 
    "Pressure (millibars)", 
]
TARGET_COLS = ["Temperature (C)", "Humidity", "Pressure (millibars)"]
MODEL_PATH = "multi_output_weather_lstm.keras"
DATA_PATH = "weatherHistory.csv" # The path used in the environment

# --- Caching Functions for Performance ---

@st.cache_data
def load_data_and_scaler(data_path, features):
    """Loads, cleans, and scales the historical weather data."""
    st.info("Loading and preprocessing data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}. Please ensure it is uploaded.")
        return None, None, None

    # Step 4: Convert Date and Clean
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df = df.sort_values("Formatted Date")
    df = df.set_index("Formatted Date")
    df = df.drop(columns=["Summary", "Daily Summary", "Loud Cover", "Apparent Temperature (C)"], errors="ignore")
    df.dropna(inplace=True) 

    # Step 4: Encoding Precip Type (Need to drop or handle if not used in FEATURES)
    # Since Precip Type is no longer a feature, we ensure it's removed or handled if needed by other steps.
    if "Precip Type" in df.columns:
        df = df.drop(columns=["Precip Type"])
    
    # Step 6: Feature Selection
    # Ensure all selected features exist after cleaning
    try:
        df_processed = df[features].copy()
    except KeyError as e:
        st.error(f"Missing feature after cleanup: {e}. Please ensure the feature exists in the original data or update the FEATURES list.")
        return None, None, None

    # Step 7: Normalization (Fit scaler on entire processed data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_processed)
    
    # Create a DataFrame for easy handling
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df_processed.index)
    
    st.success("Data loaded and scaler initialized.")
    return df_processed, scaler, scaled_df

@st.cache_resource
def load_trained_model(model_path):
    """Loads the pre-trained Keras model."""
    st.info("Loading pre-trained LSTM model...")
    try:
        model = keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Error: {e}")
        return None

# --- Prediction Logic ---

def make_prediction(model, scaler, sequence):
    """Scales, reshapes, predicts, and inverse-transforms the result."""
    if len(sequence) != SEQ_LEN:
        st.error(f"Input sequence must contain exactly {SEQ_LEN} time steps.")
        return None

    # 1. Scale input sequence
    # Use only the feature columns for scaling
    sequence_array = sequence[FEATURES].values
    scaled_input = scaler.transform(sequence_array)
    
    # 2. Reshape for LSTM (Samples, Time Steps, Features)
    # Note: Input shape is now (1, 24, 3)
    X_input = scaled_input[np.newaxis, :, :] 

    # 3. Predict
    prediction_scaled = model.predict(X_input, verbose=0) 

    # 4. Inverse Transform
    # padding_cols is now 0 (3 features - 3 targets)
    padding_cols = len(FEATURES) - len(TARGET_COLS)
    
    # Concatenate still works when padding_cols = 0
    prediction_full_array = np.concatenate((prediction_scaled, np.zeros((1, padding_cols))), axis=1)
    
    # Inverse transform and take only the first 3 columns (targets)
    prediction_inv = scaler.inverse_transform(prediction_full_array)[0, :3]
    
    # Format results
    results = pd.Series(prediction_inv, index=TARGET_COLS)
    return results

# --- Streamlit App Layout ---

st.title("🌤️ LSTM Multi-Output Weather Forecast")
st.markdown("Predicting **Temperature (°C)**, **Humidity**, and **Pressure (millibars)** for the next hour using the previous **24 hours** of data.")

# --- Load Resources ---
df_processed, scaler, scaled_df = load_data_and_scaler(DATA_PATH, FEATURES)
model = load_trained_model(MODEL_PATH)

if df_processed is None or model is None:
    st.warning("Please resolve the file loading issues to proceed.")
else:
    # --- Side Bar Controls ---
    st.sidebar.header("Prediction Controls")
    
    max_offset = len(df_processed) - SEQ_LEN - 1
    
    # Selector for the starting point of the 24-hour sequence
    start_index = st.sidebar.slider(
        "Select Historical Start Index (of 24hr sequence)",
        min_value=0, 
        max_value=max_offset,
        value=max_offset - 100 # Default to a recent point
    )

    # Identify the 24-hour input sequence
    input_sequence_df = df_processed.iloc[start_index : start_index + SEQ_LEN]
    
    # Current conditions (the last row of the input sequence)
    last_known_data = input_sequence_df.iloc[-1].copy()

    st.sidebar.subheader("Adjust Current Conditions")
    st.sidebar.markdown("Modify the last hour's data point to see a 'what if' prediction.")

    # Allow user to modify the last point of the sequence
    modified_data = {}
    
    # UI for manual input/adjustment
    for feature in FEATURES:
        # Use a consistent input type for all features
        min_val = df_processed[feature].min()
        max_val = df_processed[feature].max()
        
        # All remaining features are numerical, use number_input
        step = 0.1 
        value = st.sidebar.number_input(
            f"{feature} (Last Hour)", 
            min_value=min_val, 
            max_value=max_val, 
            value=last_known_data[feature], 
            step=step,
            key=f"input_{feature}"
        )
        modified_data[feature] = value
    
    # Create the final input sequence by replacing the last row with modified data
    final_input_sequence = input_sequence_df.copy()
    final_input_sequence.iloc[-1] = pd.Series(modified_data)
    
    # --- Main App Content: Input Data ---
    st.subheader(f"Input Data: Last {SEQ_LEN} Hours (Ending at {final_input_sequence.index[-1].strftime('%Y-%m-%d %H:%M')})")
    st.dataframe(final_input_sequence.style.format(precision=2), height=200, use_container_width=True)

    # --- Prediction Button and Results ---
    if st.button("Run Prediction for Next Hour"):
        with st.spinner(f"Predicting next hour's weather..."):
            
            # Make the prediction
            prediction_results = make_prediction(model, scaler, final_input_sequence)

            if prediction_results is not None:
                st.subheader("✅ Predicted Weather for the Next Hour")
                
                # Display Results in Metrics
                st.metric(
                    label="🌡️ Predicted Temperature", 
                    value=f"{prediction_results['Temperature (C)']:.2f} °C",
                    delta=f"{prediction_results['Temperature (C)'] - last_known_data['Temperature (C)']:.2f} °C vs Last Hour"
                )
                st.metric(
                    label="💧 Predicted Humidity", 
                    value=f"{prediction_results['Humidity']:.2f}",
                    delta=f"{prediction_results['Humidity'] - last_known_data['Humidity']:.2f} vs Last Hour"
                )
                st.metric(
                    label="⚙️ Predicted Pressure", 
                    value=f"{prediction_results['Pressure (millibars)']:.2f} mb",
                    delta=f"{prediction_results['Pressure (millibars)'] - last_known_data['Pressure (millibars)']:.2f} mb vs Last Hour"
                )
                
                # --- Plotting the Sequence and Prediction ---
                st.subheader("Predicted Temperature Trend")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Plot the 24-hour input sequence
                temp_input = final_input_sequence["Temperature (C)"].values
                ax.plot(range(SEQ_LEN), temp_input, label="Input History", color='royalblue', marker='o', markersize=4)
                
                # Plot the prediction point
                ax.plot(
                    SEQ_LEN, 
                    prediction_results['Temperature (C)'], 
                    label="Predicted Next Hour", 
                    color='red', 
                    marker='X', 
                    markersize=10
                )
                
                ax.set_title("Temperature (C) - Input Sequence and Prediction")
                ax.set_xlabel(f"Time Step (Hours) | (Hour 24 is the Prediction)")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
            else:
                st.error("Prediction failed. Check input sequence size.")

    st.markdown("---")
    st.markdown(f"**Model Details:** Uses a 2-layer LSTM model trained on {len(df_processed)} samples, using {SEQ_LEN} hours of features ({len(FEATURES)} total) to predict the next hour's key weather metrics.")
