from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Loading  the best model we have got which is random forest
model = joblib.load('final_model.pkl')

# let us Initialize the FastAPI app
app = FastAPI()

# let us mapping of encoded labels to product types
product_type_mapping = {
    0: "Coffee",
    1: "Espresso",
    2: "Herbal Tea",
    3: "Tea"
}

class PredictionRequest(BaseModel):
    data: dict  # It accept data as a dictionary where keys are feature names and values are their respective values.

#let us define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # let us convert input data into a DataFrame
    input_data = pd.DataFrame([request.data])
    
    # letus predict using the model 
    prediction = model.predict(input_data)
    
    # Mapping the numeric prediction to the product type
    predicted_product = product_type_mapping.get(prediction[0], "Unknown")
    
    return {"prediction": predicted_product}

# let us serve the model
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
