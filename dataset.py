import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import math

warnings.filterwarnings("ignore")

def retrieve_datetime(ticker, end_date):
  """
  Retrieve data from yahoo finance with additional date features (year_sin & year_cos)

  Parameter:
  - ticker = ticker of stock to retrieve, the input should be string and ends with ".jk"

  Returns:
  - data = dataframe, complete stock data with dates features
  """
  data = yf.download(ticker, end=end_date).reset_index(drop = False)
  data["Date"] = pd.to_datetime(data["Date"])
  timestamp_s = data["Date"].map(pd.Timestamp.timestamp)
  year = 365.2425 * 24 * 60 * 60
  data['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
  data['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

  return data[["Date", "Close", "year_sin", "year_cos"]]

def create_train(data, train_size, past_years = "all"):
  """
  Create train data

  Parameters:
  - data = stock dataframe
  - end_date = 1 day after the latest train data
  - past_years = number of years back the data wants to be retrieved
      Default "all" : retrieve all the data

  Returns:
  - train = dataframe, train data filtered by the date
  """
  n_data = len(data)
  end_date_index = math.ceil(train_size * n_data) - 1
  date_format = "%Y-%m-%d"
  data["Date"] = pd.to_datetime(data["Date"])
  data["date"] = data["Date"].dt.strftime(date_format)
  end_date = data["date"][end_date_index]

  if past_years == "all":
    train = data[(data["date"] < end_date)].drop(["date"], axis = 1)
  else:
    start_date = ((datetime.strptime(end_date, date_format)) - relativedelta(years = past_years)).strftime(date_format)
    train = data[(data["date"] >= start_date) & (data["date"] < end_date)].drop(["date", "Date"], axis = 1)

  return train.dropna()

def create_val_test(data, test_size, val_size, n_past = 7):
  """
  Create validation or test data

  Parameters:
  - data = stock dataframe
  - train = train dataframe
  - start_date = the first date in validation data
  - n_months = number of months for validation/test data
      Default : 1
  - n_past = number of window_data/lag_features
      Default : 7

  Returns:
  - val, test = dataframe, validation/test data filtered by the date
  """
  train_size = 1 - val_size - test_size
  train = create_train(data, train_size, past_years = "all")
  date_format = "%Y-%m-%d"
  data["Date"] = pd.to_datetime(data["Date"])
  data["date"] = data["Date"].dt.strftime(date_format)
  train["date"] = train["Date"].dt.strftime(date_format)
  n_data = len(data)
  start_date_index = data[data["date"] == train["date"].iloc[-1]].index[0] + 1
  start_date = data["date"][start_date_index]
  end_date_index = math.ceil((test_size + train_size) * n_data) - 1
  end_val_date = data["date"][end_date_index]

  val = data[(data["date"] >= start_date) & (data["date"] < end_val_date)].dropna()
  val_date = list(val["date"])
  val = val.drop(["date", "Date"], axis = 1)

  test = data[data["date"] >= end_val_date].dropna()
  test_date = list(test["date"])
  test = test.drop(["date", "Date"], axis = 1)

  train = train.drop(["date", "Date"], axis = 1)

  return train, val, test, val_date, test_date

def window_data(x_series, y_series, exog, n_past, n_future):
  X = []
  Y = []
  E = []

  for start in range(len(x_series)):
    past_end = start + n_past
    future_end = past_end + n_future

    if future_end > len(x_series):
      return np.array(X), np.array(Y), np.array(E)

    past, future, exogenous = x_series[start:past_end], y_series[past_end:future_end], exog[past_end:future_end]

    X.append(past)
    Y.append(future)
    E.append(exogenous)

def create_windowed_data(train_df, val_df, test_df, n_past = 7, n_future = 3):
  exog_cols = ["year_sin", "year_cos"]
  train_x, train_y, train_exog = window_data(train_df.drop(exog_cols, axis=1).values,
                                              train_df['Close'].values,
                                              train_df[exog_cols].values.tolist(),
                                              n_past, n_future)

  val_x, val_y, val_exog = window_data(val_df.drop(exog_cols, axis=1).values,
                                        val_df['Close'].values,
                                        val_df[exog_cols].values.tolist(),
                                        n_past, n_future)

  test_x, test_y, test_exog = window_data(test_df.drop(exog_cols, axis=1).values,
                                          test_df['Close'].values,
                                          test_df[exog_cols].values.tolist(),
                                          n_past, n_future)

  train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)
  val_y = val_y.reshape(val_y.shape[0], val_y.shape[1], 1)
  test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], 1)

  return train_x, val_x, test_x, train_y, val_y, test_y, train_exog, val_exog, test_exog