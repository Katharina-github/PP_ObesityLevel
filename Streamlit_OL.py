import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype
import shap
import matplotlib.pyplot as plt

# Pretrained scaler
scaler = StandardScaler()

scaler.mean_ = [24.25645601, 1.70163477, 86.39609754,  2.41554687, 2.69868836, 2.00865986,  1.02156931, 0.66408023]
scaler.scale_ = [6.34003659, 0.09281931, 26.42629533,  0.53643206, 0.77509335, 0.60480361,  0.85314146, 0.61211746]
scaler.n_features_in_ = len(scaler.mean_)

# Mapping textual survey inputs to numerical values
def process_survey_data(survey_data):
    mapping = {
        "FCVC": {"Never": 1, "Sometimes": 2, "Always": 3},
        "NCP": {"1-2": 1, "Three": 3, "More than three": 4},
        "CH2O": {"Less than a liter": 1, "Between 1 and 2 L": 2, "More than 2 L": 3},
        "FAF": {"I do not have": 0, "1 or 2 days": 1, "2 or 4 days": 2, "4 or 5 days": 3},
        "TUE": {"0-2 hours": 0, "3-5 hours": 1, "More than 5 hours": 2},
    }

    # Apply mapping
    survey_data["FCVC"] = mapping["FCVC"][survey_data["FCVC"]]
    survey_data["NCP"] = mapping["NCP"][survey_data["NCP"]]
    survey_data["CH2O"] = mapping["CH2O"][survey_data["CH2O"]]
    survey_data["FAF"] = mapping["FAF"][survey_data["FAF"]]
    survey_data["TUE"] = mapping["TUE"][survey_data["TUE"]]

    return survey_data

# Preprocessing function
def preprocess_inputs(user_inputs):
    # Create DataFrame from user inputs
    input_data = pd.DataFrame([user_inputs])

    # Binary encoding for categorical columns
    binary_mappings = {
        "Gender": {"Male": 1.0, "Female": 0.0},
        "family_history_with_overweight": {"Yes": 1.0, "No": 0.0},
        "FAVC" : {"Yes": 1.0, "No": 0.0},
        "SMOKE": {"Yes": 1.0, "No": 0.0},
        "SCC": {"Yes": 1.0, "No": 0.0},
    }

    for col, mapping in binary_mappings.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(mapping)


     # One-hot encoding for categorical columns (CAEC, CALC, MTRANS)
    categorical_cols = ["CAEC", "CALC", "MTRANS"]
    for col in categorical_cols:
        if col in input_data.columns:
            # Use pd.get_dummies to create one-hot encoded columns
            dummies = pd.get_dummies(input_data[col], prefix=col)
            input_data = pd.concat([input_data, dummies], axis=1)
            input_data.drop(col, axis=1, inplace=True)  # Drop the original column

    
    # Fill missing one-hot encoded columns with 0
    expected_cols = ["CAEC_Always", "CAEC_Frequently", "CAEC_Sometimes", "CAEC_no", 
                "CALC_Always", "CALC_Frequently", "CALC_Sometimes", "CALC_no",
                "MTRANS_Automobile", "MTRANS_Bike", "MTRANS_Motorbike", "MTRANS_Public_Transportation", "MTRANS_Walking"]
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0.0

    # Convert boolean columns (True/False) to 1/0
    input_data = input_data.astype(float)

    # Normalize numerical columns using the saved scaler
    numeric_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    # Ensure numerical columns exist before scaling
    for col in numeric_columns:
        if col not in input_data.columns:
            input_data[col] = 0.0

    # Transform the data
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

    # Check the data after scaling
    print(input_data[numeric_columns].describe())

    # Reorder columns to match model training
    ordered_columns = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CAEC_Always",
    "CAEC_Frequently", "CAEC_Sometimes", "CAEC_no", "CALC_Always", "CALC_Frequently",
    "CALC_Sometimes", "CALC_no", "MTRANS_Automobile", "MTRANS_Bike", "MTRANS_Motorbike",
    "MTRANS_Public_Transportation", "MTRANS_Walking"
    ]
    input_data = input_data[ordered_columns]

    return input_data

# Load the trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model(r"C:\Users\katha\Documents\PortfolioProjects\OL\xgb_obesity_model(1).json")

# Predict Obesity level function:
def predict_obesity(survey_data, xgb_model):
    # Process survey data
    processed_data = process_survey_data(survey_data)
    # Preprocess inputs
    preprocessed_data = preprocess_inputs(processed_data)
    # Convert preprocessed data to DMatrix format
    dmatrix = xgb.DMatrix(preprocessed_data)
    # Use preprocessed data for prediction
    prediction = xgb_model.predict(dmatrix)

    # Define the classe
    obesity_levels = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction)
    # Get the predicted obesity level
    predicted_class = obesity_levels[predicted_class_index]

    return predicted_class

def explain_prediction(data, model):
    """
    Explain the prediction of the XGBoost model using SHAP.
    
    Args:
        data (pd.DataFrame): Preprocessed input data for prediction.
        model (xgb.Booster): Trained XGBoost model.
    
    Returns:
        shap_values: SHAP values for the prediction.

    """
    obesity_levels = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    # Initialize SHAP explainer for the XGBoost model
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(data)

     # Predict probabilities for the classes
     # Convert preprocessed data to DMatrix format
    dmatrix = xgb.DMatrix(data)
    class_probabilities = model.predict(dmatrix)
    predicted_class_index = class_probabilities.argmax(axis=1)[0]
    predicted_class_name = obesity_levels[predicted_class_index]  # Map index to class name
    
    # Calculate SHAP values
    shap_values = explainer(data)
    
    # Plot SHAP values for the predicted class
    st.subheader(f"Explaining SHAP values for predicted class: {predicted_class_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[0, :, predicted_class_index],
            base_values=shap_values.base_values[0][predicted_class_index],
            data=data.iloc[0]
        ),
        show=False
    )
    st.pyplot(fig)

    st.subheader("Interpretation of SHAP Waterfall Graph for Predicted Class")
    st.write("The SHAP waterfall graph helps us understand how different features (the values entered in the survey) contribute to the prediction of the target class (the obesity level that was predicted), such as Overweight. Each bar represents the effect of a specific feature, with its length showing how much the feature pushes the prediction for the target class.")
    st.write("Red Bars: These features positively contribute to the predicted class (e.g., Overweight). They increase the likelihood of the participant being classified into this class.")
    st.write("Blue Bars: These features negatively contribute to the predicted class (e.g., Overweight). They decrease the likelihood of being classified into this class. However, blue bars do not directly tell you which other class the feature favors (e.g., Normal Weight or Obesity). To know this, you'd need to look at the SHAP values for those other classes (which are displayed below).")

    st.subheader(f"Explaining SHAP values for all classes")
    # Loop through all classes and visualize SHAP values
    num_classes = shap_values.values.shape[-1]  # Get number of classes
    for class_index in range(num_classes):
        class_name = obesity_levels[class_index]  # Map index to class name
        st.subheader(f"Class {class_name}:")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values.values[0, :, class_index],
                base_values=shap_values.base_values[0][class_index],
                data=data.iloc[0]
            ),
            show=False  # Prevent plt.show()
        )
        st.pyplot(fig)

# Page Config
st.set_page_config(page_title="Obesity Prediction App", layout="wide")

# Tabs
tab1, tab2, tab3, tab4= st.tabs(["Welcome", "Survey", "Results", "More Information"])

with tab1:
    st.title("Welcome to the Obesity Prediction App")
    st.write(
        """
        The aim of this app is to demonstrate how machine learning can be used in real-life scenarios. 
        By filling out the survey in the next tab, you'll receive a prediction of your weight level.

        The prediction is not only based on weight and height but also considers your eating habits and physical condition.
        
        ### Important:
        - This app is intended to showcase the usage of machine learning.
        - **It is not a substitute for professional medical advice.**
        """
    )

with tab2:
    st.title("Survey")

    # Gender
    Gender = st.radio("What is your gender?", ["Male", "Female"])
    
    # Age
    Age = st.number_input("What is your age?", min_value=14, max_value=120, step=1)
    
    # Height
    Height = st.number_input("What is your height? (in meters e.g. 1.70)", min_value=1.0, max_value=2.5, step=0.01)
    
    # Weight
    Weight = st.number_input("What is your weight? (in kilograms)", min_value=30.0, max_value=300.0, step=0.1)
    
    # Family History
    family_history_with_overweight = st.radio("Has a family member suffered or suffers from overweight?", ["Yes", "No"])
    
    # High-Caloric Food
    FAVC = st.radio("Do you eat high caloric food frequently?", ["Yes", "No"])
    
    # Vegetables
    FCVC = st.radio("Do you usually eat vegetables in your meals?", ["Never", "Sometimes", "Always"])
    
    # Main Meals
    NCP = st.radio("How many main meals do you have daily?", ["1-2", "Three", "More than three"])
    
    # Food Between Meals
    CAEC = st.radio("Do you eat any food between meals?", ["no", "Sometimes", "Frequently", "Always"])
    
    # Smoking
    SMOKE = st.radio("Do you smoke?", ["Yes", "No"])
    
    # Water Intake
    CH2O = st.radio("How much water do you drink daily?", ["Less than a liter", "Between 1 and 2 L", "More than 2 L"])
    
    # Monitor Calories
    SCC = st.radio("Do you monitor the calories you eat daily?", ["Yes", "No"])
    
    # Physical Activity
    FAF = st.radio("How often do you have physical activity?", ["I do not have", "1 or 2 days", "2 or 4 days", "4 or 5 days"])
    
    # Screen Time
    TUE = st.radio(
        "How much time do you use technological devices such as cell phone, videogames, television, computer, and others?",
        ["0-2 hours", "3-5 hours", "More than 5 hours"]
    )
    
    # Alcohol Consumption
    CALC = st.radio("How often do you drink alcohol?", ["no", "Sometimes", "Frequently", "Always"])
    
    # Transportation
    MTRANS = st.radio(
        "Which transportation do you usually use?",
        ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"]
    )

    # Submit Button
    if st.button("Submit"):
        st.session_state["survey_data"] = {
            "Gender": Gender,
            "Age": Age,
            "Height": Height,
            "Weight": Weight,
            "family_history_with_overweight": family_history_with_overweight,
            "FAVC": FAVC,
            "FCVC": FCVC,
            "NCP": NCP,
            "CAEC": CAEC,
            "SMOKE": SMOKE,
            "CH2O": CH2O,
            "SCC": SCC,
            "FAF": FAF,
            "TUE": TUE,
            "CALC": CALC,
            "MTRANS": MTRANS,
        }
        st.success("Survey submitted! Go to the Results tab to see the predictions.")

        # Add a button that will jump to the top of the page
        if st.button("Back to Top"):
            # Adding custom JavaScript to scroll to the top of the page
            st.components.v1.html("""
            <script>
                 window.scrollTo(0, 0);
            </script>
             """, height=0)


with tab3:
    st.title("Results")

    if "survey_data" in st.session_state:
        data = st.session_state["survey_data"]
        
        # Display user input
        #st.write("### Your Inputs:")
        #for key, value in data.items():
            #st.write(f"- **{key.capitalize()}:** {value}")

        # Use a copy of the data for processing
        survey_data_copy_model = st.session_state["survey_data"].copy()
        survey_data_copy_shap = st.session_state["survey_data"].copy()
        
        # Make prediction and display result
        prediction = predict_obesity(survey_data_copy_model, xgb_model)

        print("Predicted Obesity level:", prediction)

        st.write("### Predicted Weight Level:")
        st.write(f"Using the XGBoost model, your predicted weight level is {prediction}.")
        
        # Real Calculation
        st.write("### Real Weight Level (BMI):")
        bmi = data["Weight"] / (data["Height"] ** 2)
        if bmi < 18.5:
            st.write(f"Underweight (BMI: {bmi:.2f})")
        elif 18.5 <= bmi < 24.9:
            st.write(f"Normal weight (BMI: {bmi:.2f})")
        elif 25.0 <= bmi < 29.9:
            st.write(f"Overweight (BMI: {bmi:.2f})")
        elif 30.0 <= bmi < 34.9:
            st.write(f"Obesity I (BMI: {bmi:.2f})")
        elif 35.0 <= bmi < 39.9:
            st.write(f"Obesity II (BMI: {bmi:.2f})")
        else:  # BMI >= 40
            st.write(f"Obesity III (BMI: {bmi:.2f})")

        # Process survey data
        processed_data = process_survey_data(survey_data_copy_shap)
        # Preprocess inputs
        preprocessed_data = preprocess_inputs(processed_data)

        # Use SHAP to explain the prediction
        explain_prediction(preprocessed_data, xgb_model)

    else:
        st.warning("Please complete the survey in the second tab first.")


with tab4:
    st.title("More about the Obesity Prediction App")
    st.write(
        """
        Here more information of the used machine learning technics are presented.

        All results are based on the Estimation of Obesity Levels Based On Eating Habits and Physical Condition dataset: https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition?
        
        Link to the GitHub repository including all code (EDA, model selection, Streamlit presentation): https://github.com/Katharina-github/PP_ObesityLevel
        """
    )
