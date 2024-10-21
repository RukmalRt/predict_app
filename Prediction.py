import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load and preprocess the data
df = pd.read_csv('melbourne.csv')

# Encoding categorical columns with LabelEncoder
label_cols = ['Suburb', 'Type', 'Regionname']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Extract year and month from the Date column
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month

# Select relevant columns and drop rows with missing values
df_new = df[['Year', 'Month', 'Price', 'Rooms', 'Landsize', 'Bathroom', 'BuildingArea', 'YearBuilt', 'Type', 'Regionname', 'Distance']]
df_new = df_new.dropna(how='any')

# Split data into features (X) and target (y)
X = df_new.drop(columns=['Price'])
y = df_new['Price']

# Scale the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def predict_price(year, month, rooms, landsize, bathroom, building_area, year_built, house_type, region_name,distance):
    # Create a dictionary with input values
    input_data = {
        'Year': year,
        'Month': month,
        'Rooms': rooms,
        'Landsize': landsize,
        'Bathroom': bathroom,
        'BuildingArea': building_area,
        'YearBuilt': year_built,
        'Type': house_type,
        'Regionname': region_name,
        'Distance': distance
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using the LabelEncoder fitted earlier
    input_df['Type'] = le_dict['Type'].transform(input_df['Type'])
    input_df['Regionname'] = le_dict['Regionname'].transform(input_df['Regionname'])

    # Scale the features using the StandardScaler fitted earlier
    input_scaled = scaler.transform(input_df)

    # Use the trained model to predict the price
    predicted_price = model.predict(input_scaled)

    return predicted_price[0]

st.title("House Price Prediction App")

# User inputs
year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
landsize = st.number_input("Land Size (sqm)", min_value=0, value=500)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
building_area = st.number_input("Building Area (sqm)", min_value=0, value=150)
year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=1990)
house_type = st.selectbox("House Type", ['h', 'u', 't'])  # h: house, u: unit, t: townhouse
region_name = st.selectbox("Region Name", ['Northern Metropolitan', 'Western Metropolitan',
       'Southern Metropolitan', 'Eastern Metropolitan',
       'South-Eastern Metropolitan', 'Northern Victoria',
       'Eastern Victoria', 'Western Victoria'])
distance = st.number_input("Distance to City (km)", min_value=0, value=10)

# Button for prediction
if st.button("Predict Price"):
    predicted_price = predict_price(year, month, rooms, landsize, bathroom, building_area, year_built, house_type, region_name, distance)
    st.write(f"The predicted house price is: ${predicted_price:.2f}")
