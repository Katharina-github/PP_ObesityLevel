import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype
import shap
import matplotlib.pyplot as plt
import time

# to navigate back to top
js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = 0;
</script>
'''

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
    fig, ax = plt.subplots(figsize=(8, 4))
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
    st.write("For example, if Weight is red and has a large value, it strongly contributes to the prediction of Overweight. Similarly, if Age is blue, it means age lowers the likelihood of Overweight, though it may favor another class.")
    
    #Adding an explantion of the feature names
    # Define the data
    feature_data = {
        "Variable Name": [
            "Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", 
            "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
        ],
        
        "Description": [
            "Gender", "Age", "Height", "Weight", "Has a family member suffered or suffers from overweight?", 
            "Do you eat high caloric food frequently?", "How much water do you drink daily?", 
            "Do you monitor the calories you eat daily?", "How often do you have physical activity?", 
            "How much time do you use technological devices such as cell phone, videogames, television, computer and others?", 
            "How often do you drink alcohol?", "Which transportation do you usually use?", 
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(feature_data)

    # Streamlit app
    st.title("Variable Information Table")
    st.write("Below is the table containing variable information:")

    # Display the table in Streamlit
    st.dataframe(df)  # Interactive table


def deep_explain_prediction(data, model):
    """
    Explain the prediction of the XGBoost model using SHAP for all classes.
    
    Args:
        data (pd.DataFrame): Preprocessed input data for prediction.
        model (xgb.Booster): Trained XGBoost model.
    """
    obesity_levels = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    # Initialize SHAP explainer for the XGBoost model
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(data)
    
    # Calculate SHAP values
    shap_values = explainer(data)

    #Explaining text
    st.subheader("Interpreting SHAP Graphs for All Classes")
        
    st.write("### What These Graphs Show:")
    st.write("- Each graph represents how features impact the prediction probability for a specific class.")
    st.write("- For each feature:")
    st.write("  - **Red bars**: Increase the likelihood of the class.")
    st.write("  - **Blue bars**: Decrease the likelihood of the class.")
    st.write("- Unlike the single-class waterfall graph, these graphs provide a more comprehensive view of how features affect all possible classes.")
        
    st.write("### Key Insights:")
    st.write("- By comparing the graphs, you can see how the same feature might favor one class while reducing the likelihood of another.")
    st.write("- **Example:** The entered value for 'Weight' might have a strong positive impact on 'Obesity_Type_I' but negatively impact 'Normal_Weight'.")

    st.subheader(f"Explaining SHAP values for all classes")
    # Loop through all classes and visualize SHAP values
    num_classes = shap_values.values.shape[-1]  # Get number of classes
    for class_index in range(num_classes):
        class_name = obesity_levels[class_index]  # Map index to class name
        st.subheader(f"Class {class_name}:")
        fig, ax = plt.subplots(figsize=(8, 4))
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
tab1, tab2, tab3, tab4, tab5= st.tabs(["Welcome", "Survey", "Personal Results", "Model Insights", "About"])

with tab1:
    st.title("Welcome to the Obesity Prediction App")
    st.write(
        """
        The aim of this app is to demonstrate how machine learning can be used in real-life scenarios. 
        By filling out the survey in the next tab, you'll receive a prediction of your weight level.

        The prediction is not only based on weight and height but also considers your eating habits and physical condition. Additionally, the app provides an interpretation of the most important features influencing the predictions, offering insights into how these factors impact weight levels.

        Based on the data in this dataset, the app also provides actionable advice for achieving or maintaining a healthier weight.

        This app is a tool to help you reflect on your habits and make informed decisions for a healthier lifestyle.
        
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

    if st.button("Back to top", key="back_to_top_1"):
        temp = st.empty()
        with temp:
            st.components.v1.html(js)
            time.sleep(.5) # To make sure the script can execute before being deleted
        temp.empty()


with tab3:
    st.title("Personal Results")

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

        if st.button("Back to top", key="back_to_top_4"):
            temp = st.empty()
            with temp:
                st.components.v1.html(js)
                time.sleep(.5) # To make sure the script can execute before being deleted
            temp.empty()

        #Add a button for advanced users
        # Main button for advanced insights
        if st.button("Advanced Insights"):
            deep_explain_prediction(preprocessed_data, xgb_model)
            # Add back to top button
            if st.button("Back to top", key="back_to_top_2"):
                temp = st.empty()
                with temp:
                    st.components.v1.html(js)
                    time.sleep(.5) # To make sure the script can execute before being deleted
                temp.empty()


    else:
        st.warning("Please complete the survey in the second tab first.")

with tab4:
    st.title("Model Insights")
    st.write("This tab presents insights into the model's behavior and explains feature importance.")
    st.subheader("Feature Importance of the Predictive Model")
    st.image(r"C:\Users\katha\Documents\PortfolioProjects\OL\Feature_importance_XGBoost.png")
    st.write("""
        The chart above highlights the top 10 most important features that influence the model's predictions for obesity levels. Below is a detailed interpretation of each feature:

        1. **FCVC (Frequency of Vegetable Consumption)**: 
        - Regular consumption of vegetables is generally associated with a reduced risk of obesity. However, the data shows that individuals across all obesity levels, including those with "Obesity_Type_I," "Obesity_Type_II," and "Obesity_Type_III," report relatively high vegetable consumption. This suggests that while vegetable consumption is beneficial, it may not be sufficient to offset the impact of other dietary or lifestyle factors for those with obesity.

        2. **Gender**: 
        - Gender differences significantly affect the likelihood of obesity, potentially due to biological, hormonal, and behavioral factors that differ between men and women.

        3. **Weight**: 
        - Unsurprisingly, weight is one of the strongest predictors of obesity levels, as it directly contributes to Body Mass Index (BMI) calculations, which are a key metric for categorizing obesity.

        4. **CAEC (Eating Between Meals - Frequently)**: 
        - Frequent consumption of food between meals is generally associated with higher obesity levels due to increased calorie intake. However, in this dataset, frequent snacking is more strongly associated with individuals who have normal or insufficient weight. This could indicate that those with lower weight tend to snack to meet their caloric needs or that snacking behavior alone is not sufficient to predict obesity in this population.

        5. **MTRANS (Walking)**: 
        - Walking as a primary mode of transportation or physical activity is associated with a lower likelihood of obesity. Regular walking helps burn calories and maintain a healthy weight.

        6. **Family History with Overweight**: 
        - A family history of being overweight indicates the potential influence of genetic predisposition or shared lifestyle habits that contribute to obesity.

        7. **CALC (No Alcohol Consumption)**: 
        - Interestingly, the data suggests that individuals who do not consume alcohol are less likely to have obesity. This could be due to a healthier lifestyle overall or the absence of excess calorie intake from alcoholic beverages.

        8. **FAVC (High-Calorie Food Consumption)**: 
        - Consuming high-calorie foods is a significant predictor of obesity. Diets high in energy-dense foods contribute to excessive calorie intake and weight gain.

        9. **Height**: 
        - Height, as part of BMI calculations, plays a role in predicting obesity levels. Shorter individuals may have higher BMI values for the same weight, influencing the predictions.

        10. **CAEC (Eating Between Meals - Sometimes)**: 
        - Occasional snacking is common across all weight categories, including those with normal weight but more commonly for classes with higher obesity levels.

        Overall, these features collectively provide valuable insights into the factors that influence obesity levels. While individual behaviors, such as eating habits and physical activity, are important, genetic and lifestyle factors also play a significant role.
        """)

    st.write("")
    st.subheader("Final Suggestions for Weight Reduction Based on the Dataset")
    st.write("""
        Based on the findings from this dataset and supported by common sense, here are some practical suggestions to help reduce/maintaine weight:

        1. **Increase Vegetable Consumption**:
        - The dataset shows that frequent consumption of vegetables (FCVC) is associated with lower obesity levels. Including more vegetables in your diet can help you feel full while providing essential nutrients with fewer calories.

        2. **Choose Walking as a Mode of Transportation**:
        - Walking regularly, whether as a form of exercise or as a means of transportation (e.g., walking to work or errands), is associated with lower obesity levels. If possible, opt for walking instead of driving or using public transportation for shorter distances.

        3. **Limit or Avoid Alcohol Consumption**:
        - The dataset indicates that individuals who do not consume alcohol are less likely to have obesity. Alcohol is calorie-dense and often consumed in addition to regular meals, contributing to weight gain. Reducing or eliminating alcohol intake can have a positive impact on weight management.

        4. **Reduce Consumption of High-Calorie Foods**:
        - Consuming high-calorie foods (FAVC) is strongly associated with higher obesity levels. Aim to limit the intake of foods high in sugars and fats, such as fast food, sugary snacks, and fried items. Instead, focus on whole foods like fruits, vegetables, lean proteins, and whole grains.

        While the model also relies on snacking behavior (CAEC), its pattern in this data is inconsistent and does not align with general dietary recommendations. Therefore, it is not included in this list. Instead, focusing on overall healthy eating habits, physical activity, and avoiding calorie-dense foods is a more reliable approach to weight management.

        By adopting these lifestyle adjustments, you can take meaningful steps towards achieving and maintaining a healthier weight, or simply enhancing your overall well-being.
        """)

    if st.button("Back to top", key="back_to_top_3"):
        temp = st.empty()
        with temp:
            st.components.v1.html(js)
            time.sleep(.5) # To make sure the script can execute before being deleted
        temp.empty()

with tab5:
    st.title("More about the Obesity Prediction App")
    st.markdown("#### Dataset:")
    st.write(
        """
        All results are based on the **Estimation of Obesity Levels Based on Eating Habits and Physical Condition** dataset.  
        You can explore the dataset here:
        [Estimation of Obesity Levels Dataset](https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition?)
        """
    )
    st.markdown("#### Code and Implementation:")
    st.write(
        """
        The complete project, including Exploratory Data Analysis (EDA), model selection, and Streamlit presentation, is available on GitHub.  
        Visit the repository to access all the code and resources:  
        [GitHub Repository - Obesity Level Prediction](https://github.com/Katharina-github/PP_ObesityLevel)
        """
    )

