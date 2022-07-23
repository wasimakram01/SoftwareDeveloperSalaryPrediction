import streamlit as st
import pickle
import numpy as np
 


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor2 = data["model"]
transformer = data["transformer"]
 





def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")
     
    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
         
         
    )


    education = (
        "Less than a Bachelor's",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )


    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", education)
     
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        x = np.array([[country, education, experience]])
        x = transformer.transform(x)

        salary = regressor2.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")


  

        