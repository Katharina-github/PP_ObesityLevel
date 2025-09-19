# Obesity Level Prediction Project

This project uses machine learning to predict obesity levels based on eating habits and physical conditions. It offers an interactive Streamlit app for visualizing the predictions and interpreting SHAP values, providing insights into how various features influence obesity classification.

---

## Table of Contents

1. Project Overview

2. Dataset

    - Features

3. Machine Learning Techniques

4. Streamlit App

    - Installation

    - Usage

5. Repository Structure

6. Acknowledgments

---

## Project Overview

The Obesity Level Prediction project is designed to:

Predict obesity levels (e.g., Normal Weight, Overweight, Obesity Types I-III) based on lifestyle and demographic data.
Provide interpretability through SHAP values, helping users understand feature contributions to predictions.
Serve as an educational tool to explore the relationship between eating habits, physical conditions, and obesity levels.

---

## Dataset

The project is based on the Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset from the UCI Machine Learning Repository.

### Key Details:
Number of Features: 17

Target Variable: Obesity Level (7 classes)

### Features

Gender	                        Gender

Age                         	Age

Height	                        Height

Weight	                        Weight

family_history_with_overweight	Has a family member suffered or suffers from overweight?

FAVC	                        Do you eat high caloric food frequently?

CH2O	                        How much water do you drink daily?

SCC	                            Do you monitor the calories you eat daily?

FAF	                            How often do you have physical activity?

TUE                          	How much time do you use technological devices such as cell phone, videogames, television, computer and others?

CALC	                        How often do you drink alcohol?

MTRANS	                        Which transportation do you usually use?

---

## Machine Learning Techniques

Exploratory Data Analysis (EDA): Uncover trends and patterns in the data.
Model Selection: XGBoost classifier was chosen for its robust performance with tabular data.
SHAP Values: Used for model interpretability, showing how features influence predictions.

---

## Streamlit App

The Streamlit app provides an interactive interface for users to:

Upload their data or use pre-loaded examples.
View predictions and explanations for obesity levels.
Explore SHAP visualizations for interpretability.

### Key Features:

Waterfall Graphs: Show feature contributions for the predicted class.
Class-Wise SHAP Graphs: Compare how features impact all obesity levels.

### Installation

Clone this repository:

    git clone https://github.com/Katharina-github/PP_ObesityLevel.git
    cd PP_ObesityLevel

Install dependencies:

    pip install -r requirements.txt

Run the Streamlit app:

        streamlit run Streamlit_OL.py

### Usage

Open the Streamlit app in your browser.
Select or upload data to make predictions.
Explore the interactive SHAP visualizations to interpret results.
Refer to the Advanced Insights section for deeper analysis.

---

## Repository Structure

PP_ObesityLevel/

│

├── Streamlit_OL.py            # Main Streamlit app script

├── PP_RiskOfObesity.ipynb     # Notebook for exploratory data analysis, model training and evaluation script

├── requirements.txt           # Python dependencies

├── images                     # different image files

├── xgb_obesity_model(1).json  # XGBoost Machine Learning model

└── README.md                  # Project documentation

---

## Acknowledgments

Dataset Source: UCI Machine Learning Repository.
SHAP Library: Used for model interpretability.

This project was developed to combine machine learning with interpretability, fostering a better understanding of obesity levels through data.

