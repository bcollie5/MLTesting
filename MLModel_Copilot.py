# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
 
# --- 1. Data Loading and Preparation ---
# We'll use the NYC TLC (Taxi and Limousine Commission) dataset.
# This example will use the yellow taxi trip records.
# In a real-world scenario, you would download this data from the NYC TLC website.
# For this example, we'll create a sample DataFrame.
 
print("Starting the process...")
 
# Create a sample DataFrame representing the data
# In a real application, you would load your data here, for example:
# df = pd.read_csv('your_nyctaxi_data.csv')
data = {
    'tpep_pickup_datetime': ['2023-01-10 08:05:00', '2023-01-10 08:15:00', '2023-01-10 17:30:00', '2023-01-11 08:00:00', '2023-01-11 18:00:00'],
    'tpep_dropoff_datetime': ['2023-01-10 08:45:00', '2023-01-10 08:58:00', '2023-01-10 18:25:00', '2023-01-11 08:42:00', '2023-01-11 18:55:00'],
    'pickup_longitude': [-73.99, -73.98, -73.99, -73.985, -73.992],
    'pickup_latitude': [40.70, 40.69, 40.70, 40.695, 40.702],
    'dropoff_longitude': [-73.98, -73.985, -73.98, -73.982, -73.983],
    'dropoff_latitude': [40.75, 40.758, 40.75, 40.755, 40.757]
}
df = pd.DataFrame(data)
 
print("Sample data created.")
 
# --- 2. Feature Engineering ---
 
# Convert to datetime objects
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
 
# Calculate the trip duration in minutes
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
 
# For this specific problem, we are filtering for trips from Brooklyn to Times Square.
 
 
# Feature for the model: hour of the day
df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
 
print("Feature engineering complete.")
 
# --- 3. Model Training ---
# We'll use a Gradient Boosting model, which is great for this kind of tabular data.
 
# Define our features (X) and the target (y)
features = ['hour_of_day', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
target = 'duration'
 
X = df[features]
y = df[target]
 
# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
 
print("Training the model...")
 
# Train the model
model.fit(X_train, y_train)
 
print("Model training complete.")
 
# --- 4. Model Evaluation ---
# Let's see how well our model performs on the test data.
 
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
 
print(f"Model evaluation complete. Root Mean Squared Error: {rmse:.2f} minutes")
 
# --- 5. Saving the Model ---
# We'll save our trained model to a file using joblib.
 
model_filename = 'nyc_taxi_time_predictor.joblib'
joblib.dump(model, model_filename)
 
print(f"Model saved as '{model_filename}'")

# Deploy to copilot
class Input(BaseModel):
    message:str

cp_model = FastAPI()

@cp_model.post("/predict")
def predict(input: Input):
    prediction = model.predict([input.message])
    return {"result": prediction[0]}
 
# --- 6. Example Prediction ---
# Let's predict the time for a new trip from Brooklyn to Times Square.
 
'''new_trip_data = {
    'hour_of_day': [17],
    'pickup_longitude': [-73.988], # Example Brooklyn coordinates
    'pickup_latitude': [40.695],
    'dropoff_longitude': [-73.985], # Example Times Square coordinates
    'dropoff_latitude': [40.758]
}
new_trip_df = pd.DataFrame(new_trip_data)
 
loaded_model = joblib.load(model_filename)
predicted_duration = loaded_model.predict(new_trip_df)
 
print(f"\nPredicted trip duration for a new ride: {predicted_duration[0]:.2f} minutes")'''