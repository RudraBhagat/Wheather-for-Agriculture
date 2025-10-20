🌦️ Multi-Output LSTM Weather Forecasting

This project implements a deep learning-based weather forecasting model using a Long Short-Term Memory (LSTM) network. It predicts multiple weather attributes — Temperature (°C), Humidity (%), and Pressure (millibars) — simultaneously and provides an interactive Streamlit dashboard for real-time predictions.

✨ Features

🔮 Multi-Output Forecasting — Predicts temperature, humidity, and pressure simultaneously.

📈 Deep Learning Architecture — Sequential two-layer LSTM network with dense output.

⚡ Lightweight and Fast — Optimized for fast training and inference on CPUs/GPUs.

💻 Streamlit UI — Interactive input sliders and real-time weather predictions.

📊 Real Data-Driven — Trained on weatherHistory.csv dataset containing real historical weather records.

🧱 Project Structure

Weather-for-Agriculture/
│
├── app.py                           # Streamlit application for predictions
├── model_training.ipynb             # Jupyter Notebook for model training
├── multi_output_weather_lstm.keras  # Trained Keras model file
├── weatherHistory.csv               # Historical weather dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation


⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone [https://github.com/RudraBhagat/Wheather-for-Agriculture.git](https://github.com/RudraBhagat/Wheather-for-Agriculture.git)
cd Wheather-for-Agriculture


2️⃣ Create a Virtual Environment

python -m venv env
# For Windows
env\Scripts\activate          
# For Mac/Linux
source env/bin/activate       


3️⃣ Install Dependencies

Install the required Python packages either using the provided requirements.txt file or by manually listing them:

pip install -r requirements.txt
# Alternatively, install manually:
# pip install streamlit pandas numpy tensorflow scikit-learn matplotlib


🧠 Model Architecture

Layer (Type)

Output Shape

Parameters

LSTM (lstm_4)

(None, 24, 64)

17,408

LSTM (lstm_5)

(None, 32)

12,416

Dense (dense_2)

(None, 3)

99

Total Parameters



29,923

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Input Features: Temperature (°C), Humidity (%), Pressure (mb)

Output: Next-hour prediction for Temperature, Humidity, and Pressure

▶️ Run the Streamlit App

To launch the interactive weather forecasting interface:

streamlit run app.py


Then, open your browser and navigate to the local address (typically): http://localhost:8501/

📈 Workflow

Load the pre-trained model (multi_output_weather_lstm.keras).

Accept user inputs for the past 24 hours of weather data (or select a historical starting point).

Preprocess and feed the sequence into the LSTM.

Generate multi-output predictions for the next hour.

Display results visually in Streamlit.

📊 Example Output

Input: Last 24 hours of Temperature, Humidity, and Pressure data.

Output: Predicted values for next hour’s:

🌡️ Temperature (°C)

💧 Humidity (%)

🌬️ Pressure (mb)

🧩 Tech Stack

🧠 TensorFlow / Keras — Deep learning framework

🌐 Streamlit — User interface for predictions

📊 Pandas / NumPy — Data preprocessing

🧮 Scikit-learn — Scaling and evaluation

🎨 Matplotlib — Visualization

🚀 Future Enhancements

Include rainfall, wind speed, and visibility predictions

Extend forecast to 3-hour or 6-hour ahead intervals

Integrate real-time weather APIs (e.g., OpenWeather)

Deploy the model globally via Streamlit Cloud or Hugging Face Spaces

👨‍💻 Author

Rudra Bhagat
