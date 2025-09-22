import joblib
import pickle

def predict_game_sales(platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales):
    """
    Predict global sales for a video game with given features.
    Args:
        platform (str): Game platform
        year (float): Release year
        genre (str): Game genre
        publisher (str): Game publisher
        na_sales (float): North America sales (in millions)
        eu_sales (float): Europe sales (in millions)
        jp_sales (float): Japan sales (in millions)
        other_sales (float): Other regions sales (in millions)
    Returns:
        float: Predicted global sales (in millions)
    """
    from data import preprocess_input
    
    with open("../model/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    scaler = joblib.load("../model/scaler.pkl")
    
    X = preprocess_input(platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales,
                        label_encoders, scaler)
    
    model = joblib.load("../model/vgsales_model.pkl")
    prediction = model.predict(X)
    
    return prediction[0]
