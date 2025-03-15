# Sentiement Analysis Project

## Overview

This project implements a **Many Classification models** using the **Logistic Regression** , **VADER**, **Random Forest** and **XGBOOST** models. The goal is to analyse Sentiment by news.  

The training Results is as the table below:
| Model   | Accuracy          | Precision         | Recall            | F1-Score          |
|---------|------------------|------------------|------------------|------------------|
| VADER   | 0.81 | 0.90 | 0.69 | 0.78 |
| Logistic| 0.85 | 0.84 | 0.86 | 0.85 |
| Random  | 0.85 | 0.84 | 0.86 | 0.85 |
| XGB     | 0.76 | 0.70 | 0.89 | 0.78 |

We can see that the **Logistic** and **Random** models are the best but since I focused on MLOPs terms in this project I didn't give the training process alot of time.


## Features

- **Data Pipeline**
  - Load data from CSV to data frame using `pandas`
  - clean data with applying many cleaning methods
  - save the cleaned data to `CSV` file
  - split the data to `train`,`test` and `val`.
  - Vectorize the data and save it as `pkl` file to use it later.
- **Trainer**
  This part is when you wanna just train models on saved data to avoid run the data pipeline from scratch
  - load Vectorized data.
  - train all models that i have mentiond.
  - save the trained model.
  - update `evaluation_results` in Data folder
- **MLflow Runer**
  This part makes you able to run the trainer and run experements on mlflow for more understanding.
  - run trainer
  - for each model run an experiment on mlflow.
- **FastAPI Deployment**
  - Deploy the trained model using FastAPI
  - Test the Sentiment Analysis via API

- **DVC**
   It's used to deploy the data and models into google drive and update it whenever you want
   feel fre tochange the DVC configuration after pulling the code.

## Dataset Description

| Column Name  | Description                                              | Example Value  |
|-------------|----------------------------------------------------------|---------------|
| news        | Abstract or short description of the news article       | "It was a long antipodean night..." |
| neg         | Negative sentiment score (1 = most negative)            | 0.059         |
| neu         | Neutral sentiment score (1 = most neutral)              | 0.878         |
| pos         | Positive sentiment score (1 = most positive)            | 0.064         |
| compound    | Compound sentiment score (-1 = most negative, 1 = most positive) | 0.8516         |
| sentiment   | Sentiment classification (POSITIVE or NEGATIVE)         | POSITIVE      |


## Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/AliAlabed1/Sentiment-Analysis.git
cd Sentiment-Analysis
py -m pip install -r requirement.txt
```

### **2️⃣ Set Up Google Service Account for DVC**

Since we use **DVC (Data Version Control)** to manage model checkpoints and dataset storage, you need to configure Google Drive as a remote storage.

#### **Creating a Google Service Account**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project or create a new one  
3. search for Google Drive API and Enable it  
4. Navigate to **IAM & Admin > Service Accounts**
5. Click **Create Service Account**
6. Fill in the details and create the account
7. Go to the **Keys** section and click **Add Key > JSON**
8. Download the JSON file (this is your service account key)
9. Rename the file to `dvc.json` and move it to the root of the project directory

### **3️⃣ Pull the model with DVC**

After adding the service account JSON key, run the following command to fetch the dataset and model weights:

```bash
dvc pull
```

### **4️⃣ Run the Project**

Navigate to the main source directory:

```bash
cd src/app
py main.py
```

you will have 3 choices as following:
- Run the data pipline and train models from scratch   
- Run MLflow Runner
- Run the app to predict 

#### **Run the data pipline and train models from scratch **

If you want to **Run the data pipline and train models from scratch**, choose the first option in the script:  
This option will load the data pipeline that is described above.  



#### **Run MLflow Runner**

If you want to **Run MLflow Runner**, choose the second option:  
This option will run trainer an MLFLOW experiments.  

### **Run the app to predict**
This will run a FASTAPI app you can do the following:  
open your browser and visit:

```
http://localhost:8000
```

Here, you can interact with the API and test the Syntiment Analysis APP.

## API Endpoints

The FastAPI application provides the following endpoints:

| Method   | Endpoint   | Description                                                                |
| -------- | ---------- | -------------------------------------------------------------------------- |
| **POST** | `/predict` | Takes a sentence returns the predicted sentiment |
| **GET**  | `/`        | Opens the home page interface                                              |


---

Feel free to contribute by opening an issue or a pull request! 🚀
