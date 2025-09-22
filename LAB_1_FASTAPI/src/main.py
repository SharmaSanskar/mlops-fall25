from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_game_sales


app = FastAPI()

class GameData(BaseModel):
    platform: str
    year: float
    genre: str
    publisher: str
    na_sales: float
    eu_sales: float
    jp_sales: float
    other_sales: float

class SalesResponse(BaseModel):
    predicted_global_sales: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=SalesResponse)
async def predict_sales(game_features: GameData):
    try:
        prediction = predict_game_sales(
            platform=game_features.platform,
            year=game_features.year,
            genre=game_features.genre,
            publisher=game_features.publisher,
            na_sales=game_features.na_sales,
            eu_sales=game_features.eu_sales,
            jp_sales=game_features.jp_sales,
            other_sales=game_features.other_sales
        )
        
        return SalesResponse(predicted_global_sales=round(float(prediction), 2))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
