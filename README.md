# Salary Prediction Using Machine Learning

## Overview
This project predicts whether an individual's annual income exceeds $50,000 based on demographic and employment data from the UCI Adult (Census Income) dataset. It demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment of an interactive web application.

## Features
- Data loading, cleaning, and preprocessing to ensure quality dataset.
- Implementation of multiple classification models: K-Nearest Neighbors, Logistic Regression, Random Forest, and XGBoost.
- Automated model selection based on accuracy performance.
- Interactive Streamlit web application for real-time income prediction using user inputs.
- Modular and reproducible codebase developed in Python.

## Technologies Used
- Python
- pandas
- scikit-learn
- xgboost
- streamlit
- Visual Studio Code (development environment)

## Dataset
- UCI Adult (Census Income) dataset  
- Source: [Kaggle - UCI Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/<your-username>/<your-repo-name>.git
    ```
2. Navigate into the project directory:
    ```
    cd <your-repo-name>
    ```
3. Install required Python libraries:
    ```
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```
    streamlit run app.py
    ```

## Usage
- Launch the Streamlit web app.
- Enter the required demographic and employment details in the sidebar.
- Click on the "Predict Income" button.
- View the predicted salary class (<=50K or >50K) along with model accuracy.

## Project Structure
- `app.py` – Main Streamlit application file.
- `adult 3.csv` – Dataset file (ensure you download and place this in the correct path).
- `requirements.txt` – Python libraries required for the project.
- `README.md` – Project documentation.

## Results
- The best-performing model achieved an accuracy of approximately **<insert your accuracy here>**.
- XGBoost classifier was selected as the optimal model based on test set performance.

## Future Work
- Explore advanced machine learning algorithms, such as deep learning models.
- Include additional feature engineering and hyperparameter tuning.
- Deploy the application to a cloud platform for wider accessibility.
- Implement model interpretability techniques like SHAP or LIME to explain predictions.

## References
- [UCI Adult (Census Income) Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://streamlit.io/)
