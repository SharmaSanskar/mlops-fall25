# Video Game Sales Prediction API - FastAPI Lab

## Overview

This lab demonstrates how to build a **Video Game Sales Prediction API** using FastAPI. The API predicts global video game sales based on platform, genre, publisher, and regional sales data.


## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   cd src
   python train.py
   ```

3. **Start the API:**
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### Health Check - `GET /`
```json
{"status": "healthy"}
```

### Predict Sales - `POST /predict`

**Input:**
```json
{
  "platform": "PS4",
  "year": 2020.0,
  "genre": "Action",
  "publisher": "Sony",
  "na_sales": 5.2,
  "eu_sales": 3.8,
  "jp_sales": 1.1,
  "other_sales": 2.3
}
```

**Output:**
```json
{
  "predicted_global_sales": 12.45
}
```

## Testing

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

## Dataset

The dataset contains video game sales data with features:
- **Platform**: Gaming platform (PS4, Xbox, PC, etc.)
- **Year**: Release year
- **Genre**: Game genre (Action, Sports, RPG, etc.)
- **Publisher**: Game publisher
- **NA_Sales**: North America sales (millions)
- **EU_Sales**: Europe sales (millions)
- **JP_Sales**: Japan sales (millions)
- **Other_Sales**: Other regions sales (millions)
- **Global_Sales**: Total sales (target variable)

## Model

Uses Random Forest Regressor to predict global sales based on the input features.

## Screenshots

<img width="1512" height="777" alt="docs" src="https://github.com/user-attachments/assets/ff7174b9-38b8-4b21-9b12-7b28613d037b" />
<img width="1130" height="943" alt="api_response" src="https://github.com/user-attachments/assets/f354ceff-07f0-4240-ac04-4b95a531266b" />
