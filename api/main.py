from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Load trained model

model = joblib.load(
    "models/churn_model.joblib"
)


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)



# Input schema

class CustomerData(BaseModel):

    Age: int

    Gender: str

    Tenure: int

    Usage_Frequency: int

    Support_Calls: int

    Payment_Delay: int

    Subscription_Type: str

    Contract_Length: str

    Total_Spend: float

    Last_Interaction: int



# Home route

@app.get("/")
def home():

    return {
        "message": "Customer Churn API is Running"
    }



# Prediction API

@app.post("/predict")
def predict(data: CustomerData):

    try:

        # Create dataframe
        input_data = pd.DataFrame(
            [
                {
                    "Age": data.Age,

                    "Gender": data.Gender,

                    "Tenure": data.Tenure,

                    "Usage Frequency": data.Usage_Frequency,

                    "Support Calls": data.Support_Calls,

                    "Payment Delay": data.Payment_Delay,

                    "Subscription Type": data.Subscription_Type,

                    "Contract Length": data.Contract_Length,

                    "Total Spend": data.Total_Spend,

                    "Last Interaction": data.Last_Interaction
                }
            ]
        )


        print("INPUT COLUMNS")
        print(input_data.columns)



        print("MODEL EXPECTS")
        print(model.feature_names_in_)



        # Prediction

        prediction = model.predict(
            input_data
        )



        if prediction[0] == 1:

            result = "Customer will Churn"

        else:

            result = "Customer will Stay"



        return {

            "prediction": int(prediction[0]),

            "result": result

        }



    except Exception as e:


        print("ERROR:")
        print(e)


        return {

            "error": str(e)

        }