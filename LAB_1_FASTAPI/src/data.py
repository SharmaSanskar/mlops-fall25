import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    """
    Load the Video Game Sales dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the video game sales dataset.
        y (numpy.ndarray): The target values (Global_Sales) of the dataset.
        label_encoders (dict): Dictionary of label encoders for categorical features.
        scaler (StandardScaler): Fitted scaler for numerical features.
    """
    df = pd.read_csv("../assets/vgsales.csv")
    df = df.dropna()
    
    features = ['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    X_df = df[features].copy()
    y = df['Global_Sales'].values
    
    label_encoders = {}
    for feature in ['Platform', 'Genre', 'Publisher']:
        le = LabelEncoder()
        X_df[feature] = le.fit_transform(X_df[feature])
        label_encoders[feature] = le
    
    scaler = StandardScaler()
    numerical_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    X_df[numerical_features] = scaler.fit_transform(X_df[numerical_features])
    
    return X_df.values, y, label_encoders, scaler

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.3, random_state=12)

def preprocess_input(platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales, 
                    label_encoders, scaler):
    """
    Preprocess input data for prediction.
    Args:
        platform (str): Game platform
        year (float): Release year
        genre (str): Game genre
        publisher (str): Game publisher
        na_sales (float): North America sales
        eu_sales (float): Europe sales
        jp_sales (float): Japan sales
        other_sales (float): Other regions sales
        label_encoders (dict): Dictionary of fitted label encoders
        scaler (StandardScaler): Fitted scaler
    Returns:
        numpy.ndarray: Preprocessed feature vector
    """
    input_data = pd.DataFrame({
        'Platform': [platform], 'Year': [year], 'Genre': [genre], 'Publisher': [publisher],
        'NA_Sales': [na_sales], 'EU_Sales': [eu_sales], 'JP_Sales': [jp_sales], 'Other_Sales': [other_sales]
    })
    
    for feature in ['Platform', 'Genre', 'Publisher']:
        if feature in label_encoders:
            try:
                input_data[feature] = label_encoders[feature].transform(input_data[feature])
            except ValueError:
                input_data[feature] = 0
    
    numerical_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    return input_data.values