import streamlit as st

# Page Config
st.set_page_config(page_title="Obesity Prediction App", layout="wide")

# Tabs
tab1, tab2, tab3 = st.tabs(["Welcome", "Survey", "Results"])

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
    gender = st.radio("What is your gender?", ["Male", "Female"])
    
    # Age
    age = st.number_input("What is your age?", min_value=1, max_value=120, step=1)
    
    # Height
    height = st.number_input("What is your height? (in meters)", min_value=0.5, max_value=2.5, step=0.01)
    
    # Weight
    weight = st.number_input("What is your weight? (in kilograms)", min_value=10.0, max_value=300.0, step=0.1)
    
    # Family History
    family_history = st.radio("Has a family member suffered or suffers from overweight?", ["Yes", "No"])
    
    # High-Caloric Food
    high_caloric_food = st.radio("Do you eat high caloric food frequently?", ["Yes", "No"])
    
    # Vegetables
    vegetables = st.radio("Do you usually eat vegetables in your meals?", ["Never", "Sometimes", "Always"])
    
    # Main Meals
    main_meals = st.radio("How many main meals do you have daily?", ["1-2", "Three", "More than three"])
    
    # Food Between Meals
    food_between_meals = st.radio("Do you eat any food between meals?", ["No", "Sometimes", "Frequently", "Always"])
    
    # Smoking
    smoking = st.radio("Do you smoke?", ["Yes", "No"])
    
    # Water Intake
    water_intake = st.radio("How much water do you drink daily?", ["Less than a liter", "Between 1 and 2 L", "More than 2 L"])
    
    # Monitor Calories
    monitor_calories = st.radio("Do you monitor the calories you eat daily?", ["Yes", "No"])
    
    # Physical Activity
    physical_activity = st.radio("How often do you have physical activity?", ["I do not have", "1 or 2 days", "2 or 4 days", "4 or 5 days"])
    
    # Screen Time
    screen_time = st.radio(
        "How much time do you use technological devices such as cell phone, videogames, television, computer, and others?",
        ["0-2 hours", "3-5 hours", "More than 5 hours"]
    )
    
    # Alcohol Consumption
    alcohol_consumption = st.radio("How often do you drink alcohol?", ["I do not drink", "Sometimes", "Frequently", "Always"])
    
    # Transportation
    transportation = st.radio(
        "Which transportation do you usually use?",
        ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"]
    )

    # Submit Button
    if st.button("Submit"):
        st.session_state["survey_data"] = {
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
            "family_history": family_history,
            "high_caloric_food": high_caloric_food,
            "vegetables": vegetables,
            "main_meals": main_meals,
            "food_between_meals": food_between_meals,
            "smoking": smoking,
            "water_intake": water_intake,
            "monitor_calories": monitor_calories,
            "physical_activity": physical_activity,
            "screen_time": screen_time,
            "alcohol_consumption": alcohol_consumption,
            "transportation": transportation,
        }
        st.success("Survey submitted! Go to the Results tab to see the predictions.")


with tab3:
    st.title("Results")

    if "survey_data" in st.session_state:
        data = st.session_state["survey_data"]
        
        # Example: Display user input
        st.write("### Your Inputs:")
        for key, value in data.items():
            st.write(f"- **{key.capitalize()}:** {value}")
        
        # Example: Add prediction using a placeholder (replace with your model code)
        st.write("### Predicted Weight Level:")
        st.write("Using the XGBoost model, your predicted weight level is **[PLACEHOLDER]**.")

        # Interpretation
        st.write("### Interpretation:")
        st.write("Based on your eating and exercise habits, your behavior is associated with **[PLACEHOLDER]** obesity level.")
        
        # Real Calculation
        st.write("### Real Weight Level (BMI):")
        bmi = data["weight"] / (data["height"] ** 2)
        if bmi < 18.5:
            st.write(f"Underweight (BMI: {bmi:.2f})")
        elif 18.5 <= bmi < 24.9:
            st.write(f"Normal weight (BMI: {bmi:.2f})")
        elif 25.0 <= bmi < 29.9:
            st.write(f"Overweight (BMI: {bmi:.2f})")
        else:
            st.write(f"Obesity (BMI: {bmi:.2f})")
    else:
        st.warning("Please complete the survey in the second tab first.")

