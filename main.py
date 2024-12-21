from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the RandomForest model pipeline
model = joblib.load('final_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Example mapping of encoded labels to product types
product_type_mapping = {
    0: "Coffee",
    1: "Espresso",
    2: "Herbal Tea",
    3: "Tea"
}

class PredictionRequest(BaseModel):
    data: dict  # Accept data as a dictionary where keys are feature names and values are their respective values.

# Define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([request.data])
    
    # Predict using the model pipeline
    prediction = model.predict(input_data)
    
    # Map the numeric prediction to the product type
    predicted_product = product_type_mapping.get(prediction[0], "Unknown")
    
    return {"prediction": predicted_product}

# Serve the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
