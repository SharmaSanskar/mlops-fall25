from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Random Forest Regressor and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    rf_regressor = RandomForestRegressor(max_depth=10, random_state=12)
    rf_regressor.fit(X_train, y_train)
    joblib.dump(rf_regressor, "../model/vgsales_model.pkl")

def save_preprocessors(label_encoders, scaler):
    """
    Save the label encoders and scaler for later use in predictions.
    Args:
        label_encoders (dict): Dictionary of fitted label encoders.
        scaler (StandardScaler): Fitted scaler.
    """
    with open("../model/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    joblib.dump(scaler, "../model/scaler.pkl")

if __name__ == "__main__":
    X, y, label_encoders, scaler = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
    save_preprocessors(label_encoders, scaler)
