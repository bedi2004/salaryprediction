import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st

# -------- DATA LOADING & CLEANING --------
@st.cache_data
def load_clean_data():
    data = pd.read_csv('/Users/Apple/Desktop/python internship/adult 3.csv')

    # Replace '?' with 'Others' safely without chained assignment warning
    for col in ['workclass', 'occupation', 'native-country']:
        data[col] = data[col].replace('?', 'Others')

    # Filter out unwanted categories
    data = data.drop_duplicates()
    data = data[~data.workclass.isin(['Without-pay', 'Never-worked'])]
    data = data[~data.education.isin(['1st-4th', '5th-6th', 'Preschool'])]

    # Drop 'education' (keep education-num)
    data = data.drop('education', axis=1)

    # Encode income target as 0/1
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

    return data

# -------- MODEL SELECTION --------
@st.cache_resource
def train_and_select_model(data):
    y = data['income']
    X = data.drop('income', axis=1)

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    models = {
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=23),
        'Random Forest': RandomForestClassifier(random_state=23),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=23)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=23)

    best_name, best_score, best_pipe = None, 0, None
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        score = accuracy_score(y_test, preds)
        if score > best_score:
            best_name = name
            best_score = score
            best_pipe = pipe

    # Retrain best pipeline on full data before returning
    best_pipe.fit(X, y)
    return best_name, best_score, best_pipe, X, num_cols, cat_cols

# -------- STREAMLIT APP --------
def main():
    st.title("Salary Prediction App")
    st.markdown("Predict whether income exceeds $50k/year using the UCI Adult dataset.")

    data = load_clean_data()
    best_name, best_acc, clf, X, num_cols, cat_cols = train_and_select_model(data)
    st.write(f"Best Model: **{best_name}** with accuracy: **{best_acc:.3f}**")

    st.sidebar.header("Enter Person Details:")
    user_input = {}

    # Numeric inputs with min, max, median for UI range and initial value
    for col in X.columns:
        if col in num_cols:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            med_val = float(X[col].median())
            user_input[col] = st.sidebar.number_input(
                label=col,
                value=med_val,
                min_value=min_val,
                max_value=max_val)
        else:
            options = sorted(X[col].unique())
            user_input[col] = st.sidebar.selectbox(label=col, options=options)

    if st.sidebar.button("Predict Income"):
        input_df = pd.DataFrame([user_input])
        pred = clf.predict(input_df)[0]
        pred_label = ">50K" if pred == 1 else "<=50K"
        st.success(f"Predicted Income Class: **{pred_label}**")

if __name__ == "__main__":
    main()
