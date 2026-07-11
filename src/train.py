import os
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "Customer-Churn-Prediction"
)
df = pd.read_csv(
    "data/customer_churn.csv"
)
print("Dataset Loaded Successfully!")

print(df.head())

if "CustomerID" in df.columns:

    df = df.drop(
        "CustomerID",
        axis=1
    )

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

categorical_columns = X.select_dtypes(
    include="object"
).columns

numerical_columns = X.select_dtypes(
    exclude="object"
).columns

print("Categorical Columns:")
print(categorical_columns)

print("Numerical Columns:")
print(numerical_columns)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )

    ],

    remainder="passthrough"
)

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )

    ]

)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
with mlflow.start_run():
    print(
        "Training Random Forest..."
    )
    model.fit(
        X_train,
        y_train
    )
    prediction = model.predict(
        X_test
    )
    accuracy = accuracy_score(
        y_test,
        prediction
    )
    print(
        "Accuracy:",
        accuracy
    )
    print(
        classification_report(
            y_test,
            prediction
        )
    )
    mlflow.log_param(
        "model",
        "RandomForest"
    )
    mlflow.log_param(
        "n_estimators",
        100
    )
    mlflow.log_metric(
        "accuracy",
        accuracy
    )
    mlflow.sklearn.log_model(
        model,
        "customer_churn_model"
    )
os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/churn_model.joblib"
)
print(
    "Model Saved Successfully!"
)