import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = yf.download('^NSEI', start='2010-01-01', end='2022-01-01', auto_adjust=False)
df = df[['Adj Close']].reset_index()
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

tr, te = train_test_split(df, test_size=0.2, shuffle=False)
X_tr, y_tr = tr[['Days']], tr['Adj Close']
X_te, y_te = te[['Days']], te['Adj Close']

m = LinearRegression()
m.fit(X_tr, y_tr)
y_pred = m.predict(X_te)
print(f"Mean Squared Error: {mean_squared_error(y_te, y_pred)}")

plt.figure(figsize=(10, 6))
plt.plot(te['Date'], y_te, label='Actual')
plt.plot(te['Date'], y_pred, label='Predicted', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.title('NIFTY 50 Time Series Forecasting - Linear Regression')
plt.legend()
plt.show()