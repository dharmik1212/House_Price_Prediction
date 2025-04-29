# House_Price_Prediction
# 🏠 House Price Prediction Web App

A machine learning-based web application that predicts house prices using an optimized **XGBoost** model. The app is built with **Gradio** for easy interaction.

---

## 📂 Project Structure

- `Housing.csv` — Dataset containing house features and prices.
- `optimized_xgb_model.pkl` — Saved trained XGBoost model.
- `minmax_X_scaler.pkl` — Feature scaler.
- `minmax_y_scaler.pkl` — Target (price) scaler.
- `label_encoders.pkl` — Encoders for categorical features.
- `app.py` — Main Python script: data preprocessing, model training, and Gradio app.

---

## ⚙️ How It Works

1. **Preprocessing**
   - Drops the `furnishingstatus` column.
   - Encodes categorical variables (`mainroad`, `guestroom`, etc.).
   - Scales numerical variables and the target price using `MinMaxScaler`.

2. **Model Training**
   - Trains an **SVM** model (for evaluation only).
   - Trains an **XGBoost Regressor** using hyperparameter tuning (`GridSearchCV`).
   - Saves the best performing model and preprocessing objects for future predictions.

3. **Gradio App**
   - Users input house details through a simple interface.
   - The app preprocesses the input, runs the prediction, and displays the estimated house price.

---

## 📋 Requirements

Make sure to install the required packages:

```bash
pip install pandas scikit-learn xgboost gradio
