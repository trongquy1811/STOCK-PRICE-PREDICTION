import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as pyplot
import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA  

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD


tickers = ['MSFT', 'IBM', 'GOOGL', 'SPY', 'DIA', '^VIX', 'JPY=X', 'GBP=X']

data = yf.download(tickers, start="2010-01-01", auto_adjust=False)

adj_close = data['Adj Close'].copy()

return_period = 5 
Y = np.log(adj_close['MSFT']).diff(return_period).shift(-return_period)
Y.name = 'MSFT_pred'

X1 = np.log(adj_close[['GOOGL', 'IBM']]).diff(return_period)
X2 = np.log(adj_close[['JPY=X', 'GBP=X']]).diff(return_period)
X2.columns = ['DEXJPUS', 'DEXUSUK']  
X3 = np.log(adj_close[['SPY', 'DIA', '^VIX']]).diff(return_period)
X3.columns = ['SP500', 'DJIA', 'VIXCLS']  
X4 = pd.concat(
    [
        np.log(adj_close['MSFT']).diff(i)
        for i in [return_period, return_period * 3,
                  return_period * 6, return_period * 12]
    ],
    axis=1
).dropna()
X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']
X = pd.concat([X1, X2, X3, X4], axis=1)

dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]

Y = dataset[Y.name]
X = dataset[X.columns]

print("Dataset head:")
print(dataset.head())
print("Dataset shape:", dataset.shape)
if len(dataset) > 0:
    cols_order = [
        'MSFT_pred',
        'GOOGL', 'IBM',
        'DEXJPUS', 'DEXUSUK',
        'SP500', 'DJIA', 'VIXCLS',
        'MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT'
    ]
    cols_order = [c for c in cols_order if c in dataset.columns]

    correlation = dataset[cols_order].corr()

    pyplot.figure(figsize=(15, 15))
    pyplot.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

    pyplot.figure(figsize=(15, 15))
    scatter_matrix(dataset[cols_order], figsize=(12, 12))
    pyplot.show()
else:
    print("Dataset rỗng, kiểm tra lại quá trình xử lý dữ liệu.")
if len(Y) > 60: 
    res = sm.tsa.seasonal_decompose(Y, period=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()
else:
    print("Không đủ dữ liệu để seasonal_decompose với period=52.")


validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))

X_train, X_test = X.iloc[0:train_size], X.iloc[train_size:]
Y_train, Y_test = Y.iloc[0:train_size], Y.iloc[train_size:]

print("Train size:", X_train.shape, "Test size:", X_test.shape)

num_folds = 10
scoring = 'neg_mean_squared_error'
seed = 7

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('MLP', MLPRegressor()))
# Boosting
models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR', GradientBoostingRegressor()))
# Bagging
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

names = []
kfold_results = []
test_results = []
train_results = []

for name, model in models:
    names.append(name)
    # K-fold
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = -1 * cross_val_score(model, X_train, Y_train,
                                      cv=kfold, scoring=scoring)
    kfold_results.append(cv_results)

    # Train
    res_model = model.fit(X_train, Y_train)
    train_result = mean_squared_error(Y_train, res_model.predict(X_train))
    train_results.append(train_result)

    # Test
    test_result = mean_squared_error(Y_test, res_model.predict(X_test))
    test_results.append(test_result)

# Boxplot K-fold
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison: K-fold results')
ax = fig.add_subplot(111)
pyplot.boxplot(kfold_results)
ax.set_xticklabels(names)
fig.set_size_inches(15, 8)
pyplot.show()

# Biểu đồ Train/Test error
fig = pyplot.figure()
ind = np.arange(len(names))  # x locations
width = 0.35
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results, width=width, label='Train Error')
pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
fig.set_size_inches(15, 8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.show()


X_train_ARIMA = X_train.loc[:, ['GOOGL', 'IBM',
                                'DEXJPUS', 'SP500',
                                'DJIA', 'VIXCLS']]
X_test_ARIMA = X_test.loc[:, ['GOOGL', 'IBM',
                              'DEXJPUS', 'SP500',
                              'DJIA', 'VIXCLS']]

modelARIMA = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=(1, 0, 0))
model_fit = modelARIMA.fit()

error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted_ARIMA = model_fit.forecast(steps=len(Y_test), exog=X_test_ARIMA)
error_Test_ARIMA = mean_squared_error(Y_test, predicted_ARIMA)

print("ARIMA (1,0,0) train MSE:", error_Training_ARIMA)
print("ARIMA (1,0,0) test MSE :", error_Test_ARIMA)


seq_len = 2  # độ dài chuỗi cho LSTM

Y_train_LSTM = np.array(Y_train)[seq_len - 1:]
Y_test_LSTM = np.array(Y_test)

X_train_LSTM = np.zeros((X_train.shape[0] + 1 - seq_len,
                         seq_len, X_train.shape[1]))
X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))

for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[
        i:X_train.shape[0] + i + 1 - seq_len, :
    ]
    X_test_LSTM[:, i, :] = np.array(X)[
        X_train.shape[0] + i - 1:X.shape[0] + i + 1 - seq_len, :
    ]

def create_LSTMmodel(learn_rate=0.01, momentum=0.0):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1],
                                    X_train_LSTM.shape[2])))
    model.add(Dense(1))
    # Keras mới dùng 'learning_rate' thay vì 'lr'
    optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer)
    return model

LSTMModel = create_LSTMmodel(learn_rate=0.01, momentum=0.0)
LSTMModel_fit = LSTMModel.fit(
    X_train_LSTM, Y_train_LSTM,
    validation_data=(X_test_LSTM, Y_test_LSTM),
    epochs=50, batch_size=72, verbose=0, shuffle=False  # có thể tăng epochs
)

pyplot.plot(LSTMModel_fit.history['loss'], label='train')
pyplot.plot(LSTMModel_fit.history['val_loss'], '--', label='test')
pyplot.legend()
pyplot.show()

error_Training_LSTM = mean_squared_error(
    Y_train_LSTM, LSTMModel.predict(X_train_LSTM)
)
predicted_LSTM = LSTMModel.predict(X_test_LSTM)
error_Test_LSTM = mean_squared_error(Y_test, predicted_LSTM)

print("LSTM train MSE:", error_Training_LSTM)
print("LSTM test MSE :", error_Test_LSTM)


test_results_extended = test_results + [error_Test_ARIMA, error_Test_LSTM]
train_results_extended = train_results + [error_Training_ARIMA, error_Training_LSTM]
names_extended = names + ["ARIMA", "LSTM"]

fig = pyplot.figure()
ind = np.arange(len(names_extended))
width = 0.35
fig.suptitle('Algorithm Comparison (including ARIMA & LSTM)')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results_extended, width=width,
           label='Train Error')
pyplot.bar(ind + width/2, test_results_extended, width=width,
           label='Test Error')
fig.set_size_inches(15, 8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names_extended)
pyplot.show()


def evaluate_arima_model(arima_order):
    modelARIMA_tmp = ARIMA(endog=Y_train,
                           exog=X_train_ARIMA,
                           order=arima_order)
    model_fit_tmp = modelARIMA_tmp.fit()
    error = mean_squared_error(Y_train, model_fit_tmp.fittedvalues)
    return error

def evaluate_models(p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(order)
                    print('ARIMA%s MSE=%.7f' % (order, mse))
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except Exception:
                    continue
    print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))
    return best_cfg, best_score

p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

best_cfg, best_score = evaluate_models(p_values, d_values, q_values)

# ARIMA tuned
modelARIMA_tuned = ARIMA(endog=Y_train,
                         exog=X_train_ARIMA,
                         order=best_cfg)
model_fit_tuned = modelARIMA_tuned.fit()
predicted_tuned = model_fit_tuned.forecast(steps=len(Y_test),
                                           exog=X_test_ARIMA)

print("Tuned ARIMA test MSE:", mean_squared_error(Y_test, predicted_tuned))

# Vẽ actual vs predicted (tuned ARIMA)
predicted_tuned = pd.Series(predicted_tuned, index=Y_test.index)

pyplot.rcParams["figure.figsize"] = (8, 5)
pyplot.plot(np.exp(Y_test).cumprod(), 'r', label='actual')
pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b--', label='predicted')
pyplot.legend()
pyplot.show()
