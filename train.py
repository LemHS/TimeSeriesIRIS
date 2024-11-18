import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import dataset
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
import keras_tuner
import math
import os
import json

warnings.filterwarnings("ignore")

def plot_result(preds, actual, y_min, y_max, dates):
    plt.figure(figsize=(10,6))
    plt.ylim(ymin=y_min, ymax = y_max)
    plt.plot(preds, label='Predicted Values',  linestyle='--')
    plt.plot(actual, label='Actual Values')
    steps = np.arange(len(dates), step=50)
    labels = [dates[i] for i in steps]
    plt.xticks(steps, labels=labels, rotation=90)

    plt.title(f'Prediction vs Actual BBCA stock')
    plt.ylabel('Close')
    plt.xlabel('Timestep')

    plt.legend()

def evaluate_model(model, test_x, test_exog, test_y, scaler_y, test_date, path, exog = True):
  if exog:
    pred = model.predict([test_x, test_exog])
  else:
    pred = model.predict(test_x)
  predicted = scaler_y.inverse_transform(pred.reshape(pred.shape[0], pred.shape[1]))
  actual = scaler_y.inverse_transform(test_y.reshape(test_y.shape[0], test_y.shape[1]))
  y_min = min(predicted.reshape(predicted.shape[0]*predicted.shape[1], 1).tolist() + actual.reshape(actual.shape[0]*actual.shape[1], 1).tolist())[0]
  y_max = max(predicted.reshape(predicted.shape[0]*predicted.shape[1], 1).tolist() + actual.reshape(actual.shape[0]*actual.shape[1], 1).tolist())[0]
  metrics = {
      "rmse" : [],
      "mae" : [],
      "mape" : []
  }
  pred_day = {
      "p1" : [],
      "p2" : [],
      "p3" : []
  }

  actual_day = {
      "a1" : [],
      "a2" : [],
      "a3" : []
  }
  for i in range(len(predicted)):
      rmse = np.sqrt(mean_squared_error(predicted[i], actual[i]))
      mae = mean_absolute_error(predicted[i], actual[i])
      mape = mean_absolute_percentage_error(predicted[i], actual[i])
      metrics["rmse"].append(rmse)
      metrics["mae"].append(mae)
      metrics["mape"].append(mape)

      pred_day["p1"].append(predicted[i][0])
      pred_day["p2"].append(predicted[i][1])
      pred_day["p3"].append(predicted[i][2])

      actual_day["a1"].append(actual[i][0])
      actual_day["a2"].append(actual[i][1])
      actual_day["a3"].append(actual[i][2])

  print("Dataframe all dates")
  display(pd.DataFrame(metrics))
  print("Metrics mean")
  metrics_mean = pd.DataFrame(metrics).mean()
  display(pd.DataFrame(metrics).mean())
  metrics_mean.to_csv(path)
  plot_result(pred_day["p1"], actual_day["a1"], y_min, y_max, test_date)
  plot_result(pred_day["p2"], actual_day["a1"], y_min, y_max, test_date)
  plot_result(pred_day["p3"], actual_day["a1"], y_min, y_max, test_date)

  return pred_day, actual_day

def plot_models(models_actual_pred, test_date):
  model_names = [model for model in models_actual_pred.keys()]
  n_pred = len(models_actual_pred[model_names[0]][0].keys())
  fig, ax = plt.subplots(nrows = n_pred, figsize=(10,6*n_pred))
  ax[0].plot(models_actual_pred[model_names[0]][1]["a1"], label="Actual")
  ax[1].plot(models_actual_pred[model_names[0]][1]["a2"], label="Actual")
  ax[2].plot(models_actual_pred[model_names[0]][1]["a3"], label="Actual")
  ax[0].set_xticks(np.arange(len(test_date)), test_date, rotation=90)
  ax[1].set_xticks(np.arange(len(test_date)), test_date, rotation=90)
  ax[2].set_xticks(np.arange(len(test_date)), test_date, rotation=90)
  for model_name in model_names:
    ax[0].plot(models_actual_pred[model_name][0]["p1"], label=model_name, linestyle='--')
    ax[1].plot(models_actual_pred[model_name][0]["p2"], label=model_name, linestyle='--')
    ax[2].plot(models_actual_pred[model_name][0]["p3"], label=model_name, linestyle='--')

  fig.tight_layout()
  for i in range(n_pred):
    ax[i].legend(loc="upper right")


def train_model(ticker, val_size, test_size, end_date, model):
    data = dataset.retrieve_datetime(ticker, end_date)
    train, val, test, val_date, test_date = dataset.create_val_test(data, val_size, test_size)
    scaler = StandardScaler()
    scaler_y = StandardScaler()
    cols = train.drop(["Close"], axis = 1).columns
    train[cols] = scaler.fit_transform(train[cols])
    val[cols] = scaler.transform(val[cols])
    test[cols] = scaler.transform(test[cols])
    train["Close"] = scaler_y.fit_transform(train[["Close"]])
    val["Close"] = scaler_y.transform(val[["Close"]])
    test["Close"] = scaler_y.transform(test[["Close"]])
    train_x, val_x, test_x, train_y, val_y, test_y, train_exog, val_exog, test_exog = dataset.create_windowed_data(train, val, test)
    model, history = model(train_x, val_x, train_y, val_y)
    pred, actual = evaluate_model(model, test_x, test_exog, test_y, scaler_y, test_date, exog = False)
    
    return model, history, pred, actual

def hypertune(ticker, val_size, test_size, end_date, model, type):
    data = dataset.retrieve_datetime(ticker, end_date)
    train, val, test, val_date, test_date = dataset.create_val_test(data, val_size, test_size)
    scaler = StandardScaler()
    scaler_y = StandardScaler()
    cols = train.drop(["Close"], axis = 1).columns
    train[cols] = scaler.fit_transform(train[cols])
    val[cols] = scaler.transform(val[cols])
    test[cols] = scaler.transform(test[cols])
    train["Close"] = scaler_y.fit_transform(train[["Close"]])
    val["Close"] = scaler_y.transform(val[["Close"]])
    test["Close"] = scaler_y.transform(test[["Close"]])
    train_x, val_x, test_x, train_y, val_y, test_y, train_exog, val_exog, test_exog = dataset.create_windowed_data(train, val, test)
    model = model(train_x, val_x, train_y, val_y, type)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-4 * math.exp(-0.1 * x))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    path = f"trials/{model.name}/Hypertune{type}{model.name}"
    n_trial = len(os.listdir(path)) + 1
    tuner = keras_tuner.RandomSearch(
        hypermodel = model,
        objective = keras_tuner.Objective("val_root_mean_squared_error", direction="min"),
        overwrite=True,
        directory=f"{path}/{n_trial}"
    )

    os.makedirs(f"{path}/{n_trial}/metrics")

    tuner.search(train_x, train_y, validation_data=(val_x, val_y), callbacks=[lr_scheduler, early_stopping], epochs=100)

    best_model = tuner.get_best_models()[0]

    best_hp = tuner.get_best_hyperparameters()[0].values

    with open(f"{path}/{n_trial}/best_hp.json", "w") as file:
       json.dump(best_hp, file)

    pred, actual = evaluate_model(best_model, test_x, test_exog, test_y, scaler_y, test_date, f"{path}/{n_trial}/metrics/metrics.csv", exog = False)

    return best_model, tuner, pred, actual, test_date