import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import joblib
import os

class StockPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.marketstack.com/v1/"
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def search_company(self, query):
        """Search for stock symbols by company name"""
        endpoint = "tickers"
        params = {
            'access_key': self.api_key,
            'search': query,
            'limit': 10
        }
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                return None, "No results found"
            
            results = []
            for item in data['data']:
                results.append({
                    'name': item.get('name', 'N/A'),
                    'symbol': item.get('symbol', 'N/A'),
                    'stock_exchange': item.get('stock_exchange', {}).get('name', 'N/A')
                })
            
            return results, None
        except Exception as e:
            return None, str(e)
        
    def get_stock_data(self, symbol, start_date, end_date):
        """Fetch historical stock data"""
        endpoint = "eod"
        params = {
            'access_key': self.api_key,
            'symbols': symbol,
            'date_from': start_date,
            'date_to': end_date,
            'limit': 1000
        }
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return None, "No data available"
                
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df, None
        except Exception as e:
            return None, str(e)
    
    def preprocess_data(self, df):
        """Prepare data for machine learning"""
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df['daily_return'] = df['close'].pct_change()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['daily_return'].rolling(5).std()
        df.dropna(inplace=True)
        
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        lookback = 10
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 3])  # Close price
            
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test, df
    
    def train_model(self, symbol, df):
        """Train Random Forest model"""
        X_train, X_test, y_train, y_test, _ = self.preprocess_data(df)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        mse = mean_squared_error(y_test, y_pred)
        
        model_path = os.path.join(self.models_dir, f"{symbol}_model.pkl")
        joblib.dump(model, model_path)
        
        scaler_path = os.path.join(self.models_dir, f"{symbol}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        return model, mse
    
    def predict_future_prices(self, symbol, days, df=None):
        """Predict future prices"""
        model_path = os.path.join(self.models_dir, f"{symbol}_model.pkl")
        scaler_path = os.path.join(self.models_dir, f"{symbol}_scaler.pkl")
        
        if not os.path.exists(model_path):
            if df is None:
                return None, "Model not found"
            model, _ = self.train_model(symbol, df)
            scaler = self.scaler
        else:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        
        if df is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df, error = self.get_stock_data(symbol, start_date, end_date)
            if error:
                return None, error
        
        _, _, _, _, processed_df = self.preprocess_data(df)
        scaled_data = scaler.transform(processed_df.tail(10))
        
        predictions = []
        current_sequence = scaled_data
        for _ in range(days):
            next_pred = model.predict(current_sequence.reshape(1, -1))
            predictions.append(next_pred[0])
            new_row = np.append(current_sequence[-1, :-1], next_pred)
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        dummy_data = np.zeros((len(predictions), processed_df.shape[1]))
        dummy_data[:, 3] = predictions
        predictions = scaler.inverse_transform(dummy_data)[:, 3]
        
        future_dates = [processed_df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
        return pd.Series(predictions, index=future_dates), None
    
    def get_available_models(self):
        """Get list of trained models"""
        return [f.replace("_model.pkl", "") for f in os.listdir(self.models_dir) 
                if f.endswith("_model.pkl")]