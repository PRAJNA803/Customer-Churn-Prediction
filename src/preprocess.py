import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/customer_churn.csv")

print("Dataset Loaded Successfully!\n")

print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.fillna(df.mean(numeric_only=True), inplace=True)

print(df["Churn"].unique())

df["Churn"] = df["Churn"].astype(str).str.strip()

df["Churn"] = df["Churn"].replace({
    "No": 0,
    "Yes": 1
})

df = df.dropna(subset=["Churn"])

df["Churn"] = df["Churn"].astype(int)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Processed files saved inside the data folder.")