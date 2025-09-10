# COVID-19 Daily Infections Prediction (Poland, 2020–2023)

This project focuses on predicting the **daily number of COVID-19 infections** in Poland between **2020 and 2023** using **time series analysis**.  
The model leverages both **Polish** and **international datasets** to improve forecasting accuracy.  
For the prediction, we used an **LSTM (Long Short-Term Memory)** neural network, which is well-suited for time series data.

---

## 🚀 Features
- Time series forecasting of COVID-19 daily infections
- LSTM-based deep learning model
- Data preprocessing and feature engineering
- Visualization of results and model evaluation
- Modular and scalable project structure

---

## 🧠 Model
We used an **LSTM** neural network implemented in **PyTorch**.  
LSTM was chosen because it captures long-term dependencies and temporal patterns effectively, which is crucial for time series forecasting.

---

## 📂 Project Structure
├── main.py # Entry point to run the project
├── data
│ ├── dataset.py # Custom dataset
│ └── data_preprocessing.py # prepares sequences for training\
├── utils
│ └── seed.py # Ensures reproducibility by setting random seeds
├── training
│ └── training.py # Handles model training, validation, and saving
├── models
│ └── model.py # Defines the LSTM architecture
├── results
│ ├── visualizations.py # Plots predictions

│ └── evaluation.py # Evaluates the model’s performance
└── README.md
