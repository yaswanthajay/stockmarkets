import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = yf.download("AAPL", period="6mo", interval="1d")
df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['macd'] = ta.trend.MACD(df['Close']).macd()
bb = ta.volatility.BollingerBands(df['Close'])
df['bollinger_h'] = bb.bollinger_hband()
df['bollinger_l'] = bb.bollinger_lband()
df['volume_avg'] = df['Volume'].rolling(window=10).mean()
df.dropna(inplace=True)

# Features
features = ['Close', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'volume_avg']
X = df[features]
y = df['Close'].shift(-1).dropna()
X = X.iloc[:-1]  # Align X and y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'stockmodel.pkl')
