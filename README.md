# Mobile Game In-App Purchase Analysis

This project analyzes in-app purchase data from a mobile game, with a focus on identifying "whale" users (high-spending players). It includes a Streamlit dashboard for interactive analysis and model explanation.

## Features

*   **Data Loading and Cleaning:** Loads the raw data and performs basic cleaning.
*   **Feature Engineering:** Creates new features like logarithmic spend and a binary "whale" indicator.
*   **Predictive Modeling:** A `HistGradientBoostingClassifier` is trained to predict which users are likely to be "whales".
*   **Interactive Dashboard:** A Streamlit application provides visualizations of the data and model performance.
*   **Model Performance:** The dashboard displays the ROC-AUC score, a ROC curve, and a confusion matrix for the model.
*   **Model Explainability:** SHAP (SHapley Additive exPlanations) is used to explain individual predictions.

## File Descriptions

*   `mobile_game_inapp_purchases.csv`: The raw dataset containing information about players and their in-app purchases.
*   `clean_game_data.parquet`: A cleaned version of the dataset in Parquet format.
*   `Analysisi.ipynb`: A Jupyter notebook for exploratory data analysis.
*   `dashboard.py`: A Streamlit application for visualizing the data and model predictions.
*   `requirements.txt`: A list of the Python libraries required to run this project.
*   `README.md`: This file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd mobile_game_inapp_purchases_analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the interactive dashboard, execute the following command in your terminal:

```bash
streamlit run dashboard.py
```

This will open the dashboard in your web browser.
