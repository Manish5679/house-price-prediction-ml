import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

model = load_model()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["Business Problem", "Data Insights", "Prediction"])

if page == "Business Problem":
    st.title("House Price Prediction Using Machine Learning")
    st.subheader("Business Problem")
    st.write("A real-estate business needs a data-driven way to estimate the selling price of houses based on property characteristics.")
    st.subheader("Analytics Objective")
    st.write("Develop a regression model that predicts house price using features such as area, bedrooms, bathrooms, stories, parking, road access, air conditioning and furnishing status.")
    st.subheader("Dataset Information")
    st.write("Source: Kaggle Housing Prices Dataset")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write("Target Variable: price")
    st.subheader("Business Decision")
    st.write("The predicted price can support property valuation, listing-price decisions and negotiations.")

elif page == "Data Insights":
    st.title("Data Insights")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["price"], bins=25)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    with col2:
        st.subheader("Area vs Price")
        fig, ax = plt.subplots()
        ax.scatter(df["area"], df["price"], alpha=0.55)
        ax.set_xlabel("Area")
        ax.set_ylabel("Price")
        st.pyplot(fig)
    st.subheader("Average Price by Furnishing Status")
    st.bar_chart(df.groupby("furnishingstatus")["price"].mean())
    st.subheader("Key Insight")
    st.write("Larger area, more bathrooms, more stories, parking availability and better furnishing status are generally associated with higher house prices.")

else:
    st.title("Predict House Price")
    st.write("Enter property details and click Predict.")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area", min_value=1000, max_value=20000, value=5000, step=100)
        bedrooms = st.selectbox("Bedrooms", sorted(df["bedrooms"].unique()))
        bathrooms = st.selectbox("Bathrooms", sorted(df["bathrooms"].unique()))
        stories = st.selectbox("Stories", sorted(df["stories"].unique()))
        parking = st.selectbox("Parking Spaces", sorted(df["parking"].unique()))
    with col2:
        mainroad = st.selectbox("Main Road Access", sorted(df["mainroad"].unique()))
        guestroom = st.selectbox("Guestroom", sorted(df["guestroom"].unique()))
        basement = st.selectbox("Basement", sorted(df["basement"].unique()))
        hotwaterheating = st.selectbox("Hot Water Heating", sorted(df["hotwaterheating"].unique()))
        airconditioning = st.selectbox("Air Conditioning", sorted(df["airconditioning"].unique()))
        prefarea = st.selectbox("Preferred Area", sorted(df["prefarea"].unique()))
        furnishingstatus = st.selectbox("Furnishing Status", sorted(df["furnishingstatus"].unique()))

    input_data = pd.DataFrame({
        "area": [area], "bedrooms": [bedrooms], "bathrooms": [bathrooms],
        "stories": [stories], "mainroad": [mainroad], "guestroom": [guestroom],
        "basement": [basement], "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning], "parking": [parking],
        "prefarea": [prefarea], "furnishingstatus": [furnishingstatus]
    })

    if st.button("Predict House Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted House Price: ₹{prediction:,.0f}")
        st.write("Recommended action: Use this estimate as a decision-support value for pricing, listing or negotiation. The final price should also consider market conditions and property-specific factors not captured in the dataset.")
