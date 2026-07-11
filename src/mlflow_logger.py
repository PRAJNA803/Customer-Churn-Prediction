import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Customer Churn Prediction")

model = joblib.load("models/churn_model.joblib")

with mlflow.start_run():

    mlflow.log_param("Model", "RandomForest")

    mlflow.log_metric("Accuracy", 0.99)

    mlflow.sklearn.log_model(model, "CustomerChurnModel")

print("MLflow Run Logged Successfully!")