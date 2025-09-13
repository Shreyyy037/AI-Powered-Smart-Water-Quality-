# AI-Powered-Smart-Water-Quality-
A Machine Learning Approach to Predict Water Potability and Ensure Public Health

# AI-Powered Smart Water Quality Dashboard

This project is an end-to-end machine learning solution designed to predict water potability based on various chemical and physical metrics. The final model is deployed as an interactive web dashboard using Streamlit, allowing users to get instant predictions on their own data.

## 📋 Table of Contents
- [Project Goal](#-project-goal)
- [Features](#-features)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Setup and Installation](#-setup-and-installation)
- [How to Use](#-how-to-use)
- [Future Improvements](#-future-improvements)

## 🎯 Project Goal
The primary objective is to build a reliable machine learning model to classify water as "potable" (safe to drink) or "not potable". The secondary goal is to create a user-friendly web interface that makes this predictive power accessible to non-technical users.

## ✨ Features
The deployed Streamlit dashboard includes the following features:
* **CSV File Upload**: Users can upload a CSV file containing water quality data.
* **Data Preview**: The application displays the first few rows of the uploaded dataset.
* **Instant Predictions**: The model predicts the potability for each sample and displays the results in a table with a "Predicted_Potability" column labeled 'Safe' or 'Unsafe'.

## ⚙️ Methodology
The model was developed following a structured machine learning workflow:

1.  **Data Exploration and Preprocessing**:
    * The `water_potability.csv` dataset was loaded and analyzed.
    * Missing values in the `ph`, `Sulfate`, and `Trihalomethanes` columns were handled using median imputation.
    * All features were scaled using `StandardScaler` to normalize their ranges.
    * The data was split into an 80% training set and a 20% testing set, using stratification to maintain class distribution.

2.  **Model Building and Evaluation**:
    * Three different classification models were trained and compared: Logistic Regression, Decision Tree, and Random Forest.
    * Techniques like `class_weight='balanced'` and SMOTE were used to address class imbalance in the dataset.
    * Hyperparameter tuning was performed using `GridSearchCV` to find the optimal parameters for the baseline models.

3.  **Model Selection**:
    * Models were evaluated based on Accuracy, Precision, Recall, and F1-Score.
    * The **Random Forest Classifier** was selected as the best-performing model.

## 📊 Model Performance
The final selected model, a Random Forest Classifier, achieved the following performance on the unseen test data:
* **Accuracy**: ~66%
* **Precision (Potable class)**: 0.63
* **Recall (Potable class)**: 0.30
* **F1-Score (Potable class)**: 0.41

The model significantly outperformed the initial baselines, providing a more robust solution for prediction.

## 💻 Technology Stack
* **Language**: Python
* **Libraries**:
    * Scikit-learn (for modeling, preprocessing, and evaluation)
    * Pandas (for data manipulation)
    * Streamlit (for the web dashboard)
    * Joblib (for saving the trained model)
    * Imbalanced-learn (for SMOTE)

## 🛠️ Setup and Installation
To run this project on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone https://<your-repository-url>
    cd <repository-name>
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    It is recommended to create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    pandas
    scikit-learn
    joblib
    ```
    Then, install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Use
1.  Ensure that `app.py` and your trained model file, `best_model.pkl`, are in the same directory.
2.  Open your terminal or command prompt and navigate to the project directory.
3.  Activate your virtual environment if you created one.
4.  Run the Streamlit application with the following command:
    ```bash
    streamlit run app.py
    ```
5.  Your web browser will open a new tab with the dashboard.
6.  Click "Browse files" to upload your CSV data and view the predictions.

## 🔮 Future Improvements
* **Improve Model Accuracy**: Experiment with more powerful ensemble models like XGBoost or LightGBM.
* **Tune Final Model**: Perform extensive hyperparameter tuning on the final Random Forest model to further optimize its performance.
* **Enhance Dashboard**: Add more interactive visualizations, prediction summaries, and options for manual data entry via sliders or input fields.
