# Customer-Churn-Prediction

## Project Overview
Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a company based on customer information. It helps businesses identify customers who are at risk of churning so they can take preventive actions.

## Features
- Customer churn prediction using Machine Learning
- Data preprocessing and feature engineering
- Random Forest Classifier model
- FastAPI REST API
- Docker support for containerization
- MLflow for experiment tracking

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- MLflow
- Joblib
- Docker
- Git & GitHub

## Project Structure

Customer-Churn-Prediction/
├── api/
├── data/
├── models/
├── src/
├── Dockerfile
├── requirements.txt
└── README.md

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

2. Create a virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

## Run the API

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Future Enhancements

- Improve model accuracy
- Deploy on AWS
- Add a web interface
- Integrate with a database

## Author

Prajna Acharya
BE – Computer Science and Engineering (Data Science)
