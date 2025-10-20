# 🌦️ Multi-Output LSTM Weather Forecasting

This project implements a **deep learning-based weather forecasting model** using a **Long Short-Term Memory (LSTM)** network. It predicts multiple weather attributes — **Temperature (°C)**, **Humidity (%)**, and **Pressure (millibars)** — simultaneously and provides an interactive **Streamlit dashboard** for real-time predictions.

---

## ✨ Features

* 🔮 **Multi-Output Forecasting** — Predicts temperature, humidity, and pressure simultaneously.  
* 📈 **Deep Learning Architecture** — Sequential two-layer LSTM network with dense output.  
* ⚡ **Lightweight and Fast** — Optimized for fast training and inference on CPUs/GPUs.  
* 💻 **Streamlit UI** — Interactive input sliders and real-time weather predictions.  
* 📊 **Real Data-Driven** — Trained on `weatherHistory.csv` dataset containing real historical weather records.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone [https://github.com/RudraBhagat/Wheather-for-Agriculture.git](https://github.com/RudraBhagat/Wheather-for-Agriculture.git)
cd Wheather-for-Agriculture

### 2️⃣ **Create a Virtual Environment**
python -m venv env
env\Scripts\activate      

### 3️⃣ Install Dependencies
pip install streamlit pandas numpy tensorflow scikit-learn matplotlib

### ▶️ Run the Streamlit App
streamlit run app.py

---

### 📊 Example Output
Input: Last 24 hours of Temperature, Humidity, and Pressure data.

Output: Predicted values for next hour’s:

🌡️ Temperature (°C)

💧 Humidity (%)

🌬️ Pressure (mb)

---

### 🧩 **Tech Stack**
*🧠 **TensorFlow / Keras** — Deep learning framework

*🌐 **Streamlit** — User interface for predictions

*📊 **Pandas / NumPy** — Data preprocessing

*🧮 **Scikit-learn** — Scaling and evaluation

*🎨 **Matplotlib** — Visualization

---

### 👨‍💻 **Author**
Rudra Bhagat
