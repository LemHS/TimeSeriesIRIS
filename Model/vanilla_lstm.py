import warnings
import math

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
import keras_tuner

class StackAndTransposeLayer(Layer):
    def call(self, inputs):
        stacked = tf.stack(inputs, axis=0)
        transposed = tf.transpose(stacked, perm=[1, 0, 2])
        return transposed

def SingleShotVanillaLSTM(train_x, val_x, train_y, val_y):
    """
    Create a Single Shot Multi-step LSTM Model and train it using the training & validation data.
    """

    # Initialize LSTM State and make first prediction
    input_ts = Input(shape=(train_x.shape[1], train_x.shape[2]))
    lstm_1 = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
    # Adding Dropout Layer
    dropout_1 = Dropout(0.5)(lstm_1)

    x, *state = RNN(LSTMCell(64), return_state=True)(dropout_1)

    x = Dense(32)(x)
    prediction = Dense(3)(x)
    ss_lstm = Model(inputs=input_ts, outputs=prediction)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-4 * math.exp(-0.1 * x))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    ss_lstm.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.RootMeanSquaredError()])

    history = ss_lstm.fit(train_x, train_y,
                          epochs=30,
                          validation_data=(val_x, val_y),
                          batch_size=8,
                          verbose=1,
                          callbacks=[lr_scheduler, early_stopping])

    # Save the trained model
    ss_lstm.save('model_ss.keras')

    return ss_lstm, history

def Seq2SeqVanillaLSTM(train_x, val_x, train_y, val_y):
    """
    Create an Autoregressive LSTM Model with a Dropout layer and train it using the training & validation data.
    """
    n_steps_in = train_x.shape[1]
    n_steps_out = train_y.shape[1]
    n_features = train_x.shape[2]
    seq2seq_model = Sequential()
    seq2seq_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    seq2seq_model.add(LSTM(100, activation='relu'))
    seq2seq_model.add(RepeatVector(n_steps_out))
    seq2seq_model.add(LSTM(100, activation='relu', return_sequences=True))
    seq2seq_model.add(TimeDistributed(Dense(1)))
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-4 * math.exp(-0.1 * x))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    seq2seq_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.RootMeanSquaredError()])

    history = seq2seq_model.fit(train_x, train_y,
                          epochs=30,
                          validation_data=(val_x, val_y),
                          batch_size=8,
                          verbose=1,
                          callbacks=[lr_scheduler, early_stopping])

    # Save the trained model
    seq2seq_model.save('model_seq2seq.keras')

    return seq2seq_model, history

def AutoregressiveVanillaLSTM(train_x, val_x, train_y, val_y):
    """
    Create an Autoregressive LSTM Model with a Dropout layer and train it using the training & validation data.
    """

    predictions = []

    # Initialize LSTM State and make first prediction
    input_ts = Input(shape=(train_x.shape[1], train_x.shape[2]))
    lstm_1 = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
    # lstm_1, *state = LSTM(128, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
    # Adding Dropout Layer
    dropout_1 = Dropout(0.5)(lstm_1)

    x, *state = RNN(LSTMCell(128), return_state=True)(dropout_1)

    lstm_cell = LSTMCell(128)
    dense_1 = Dense(32)
    dense_2 = Dense(1)

    x = dense_1(x)
    # x = dense_1(dropout_1)
    prediction = dense_2(x)

    predictions.append(prediction)

    for n in range(1, 3):
        x = prediction
        x, state = lstm_cell(x, states=state)
        x = dense_1(x)
        prediction = dense_2(x)
        predictions.append(prediction)

    predictions = StackAndTransposeLayer()(predictions)

    ar_lstm = Model(inputs=input_ts, outputs=predictions)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-4 * math.exp(-0.1 * x))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    ar_lstm.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.RootMeanSquaredError()])

    history = ar_lstm.fit(train_x, train_y,
                          epochs=30,
                          validation_data=(val_x, val_y),
                          batch_size=8,
                          verbose=1,
                          callbacks=[lr_scheduler, early_stopping])

    # Save the trained model
    ar_lstm.save('model_ar.keras')

    return ar_lstm, history

class BuildVanillaLSTM(keras_tuner.HyperModel):
    def __init__(self, train_x, val_x, train_y, val_y, type):
        self.train_x = train_x
        self.val_x = val_x
        self.train_y = train_y
        self.val_y = val_y
        self.type = type
        self.name = "VanillaLSTM"


    def build(self, hp):
        if self.type == 'SingleShot':
            return self.HypertuneSingleShotVanillaLSTM(hp)
        elif self.type == 'Seq2Seq':
            return self.HypertuneSeq2SeqVanillaLSTM(hp)
        elif self.type == 'Autoregressive':
            return self.HypertuneAutoregressiveVanillaLSTM(hp)
        
    def fit(self, hp, model, *args, **kwargs):
        self.batch_size = hp.Int("batch_size", min_value=4, max_value=128, step=2, sampling="log")
        return model.fit(
            *args,
            batch_size = self.batch_size,
            **kwargs
        )
    
    def HypertuneSingleShotVanillaLSTM(self, hp):
        """
        Create a Single Shot Multi-step LSTM Model with tunable hyperparameters using Keras Tuner.
        """
        # Define hyperparameters
        lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=32)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=64, max_value=256, step=32)
        dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.01, max_value=0.5, step=0.01)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        # Model architecture
        input_ts = Input(shape=(self.train_x.shape[1], self.train_x.shape[2]))
        lstm_1 = LSTM(lstm_units_1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
        dropout_1 = Dropout(dropout_rate)(lstm_1)

        x, *state = RNN(LSTMCell(lstm_units_2), return_state=True)(dropout_1)

        x = Dense(dense_units, activation='relu')(x)
        prediction = Dense(3)(x)
        single_shot_model = Model(inputs=input_ts, outputs=prediction)

        # Compile the model
        single_shot_model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.metrics.RootMeanSquaredError()],
        )

        return single_shot_model
    
    def HypertuneSeq2SeqVanillaLSTM(self, hp):
        """
        Build a tunable Seq2Seq Vanilla LSTM Model.
        """
        n_steps_in = self.train_x.shape[1]
        n_steps_out = self.train_y.shape[1]
        n_features = self.train_x.shape[2]

        # Define hyperparameters
        lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=32)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=64, max_value=256, step=32)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        # Model architecture
        seq2seq_model = Sequential()
        seq2seq_model.add(LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        seq2seq_model.add(LSTM(lstm_units_2, activation='relu'))
        seq2seq_model.add(RepeatVector(n_steps_out))
        seq2seq_model.add(LSTM(lstm_units_2, activation='relu', return_sequences=True))
        seq2seq_model.add(TimeDistributed(Dense(1)))

        # Compile the model
        seq2seq_model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.metrics.RootMeanSquaredError()]
        )

        return seq2seq_model
    
    def HypertuneAutoregressiveVanillaLSTM(self, hp):
        """
        Build a tunable Autoregressive Vanilla LSTM Model.
        """
        predictions = []

        # Hyperparameters
        lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=32)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=64, max_value=256, step=32)
        dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.01, max_value=0.5, step=0.01)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        # Input shape
        input_ts = Input(shape=(self.train_x.shape[1], self.train_x.shape[2]))

        # Initial LSTM Layer
        lstm_1 = LSTM(lstm_units_1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
        # lstm_1, *state = LSTM(lstm_units, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
        dropout_1 = Dropout(dropout_rate)(lstm_1)

        x, *state = RNN(LSTMCell(lstm_units_2), return_state=True)(dropout_1)

        # Define LSTM Cell and Dense layers
        lstm_cell = LSTMCell(lstm_units_2)
        dense_1 = Dense(dense_units, activation='relu')
        dense_2 = Dense(1)

        # First Prediction
        x = dense_1(x)
        # x = dense_1(dropout_1)
        prediction = dense_2(x)
        predictions.append(prediction)

        # Autoregressive Predictions
        for n in range(1, 3):
            x = prediction
            x, state = lstm_cell(x, states=state)
            x = dense_1(x)
            prediction = dense_2(x)
            predictions.append(prediction)

        predictions = StackAndTransposeLayer()(predictions)

        # Build Model
        ar_model = Model(inputs=input_ts, outputs=predictions)

        # Compile Model
        ar_model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.metrics.RootMeanSquaredError()]
        )

        return ar_model

# def BuildVanillaLSTM(train_x, val_x, train_y, val_y, type):
#     def HypertuneSingleShotVanillaLSTM(hp):
#         """
#         Create a Single Shot Multi-step LSTM Model with tunable hyperparameters using Keras Tuner.
#         """
#         # Define hyperparameters
#         lstm_1_units = hp.Int('lstm_1_units', min_value=32, max_value=256, step=32)
#         lstm_2_units = hp.Int('lstm_2_units', min_value=32, max_value=128, step=16)
#         dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
#         dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
#         learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

#         # Model architecture
#         input_ts = Input(shape=(train_x.shape[1], train_x.shape[2]))
#         lstm_1 = LSTM(lstm_1_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
#         dropout_1 = Dropout(dropout_rate)(lstm_1)

#         x, *state = RNN(LSTMCell(lstm_2_units), return_state=True)(dropout_1)

#         x = Dense(dense_units, activation='relu')(x)
#         prediction = Dense(3)(x)
#         single_shot_model = Model(inputs=input_ts, outputs=prediction)

#         # Compile the model
#         single_shot_model.compile(
#             loss=tf.losses.MeanSquaredError(),
#             optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
#             metrics=[tf.metrics.RootMeanSquaredError()],
#         )

#         return single_shot_model
    
#     def HypertuneSeq2SeqVanillaLSTM(hp):
#         """
#         Build a tunable Seq2Seq Vanilla LSTM Model.
#         """
#         n_steps_in = train_x.shape[1]
#         n_steps_out = train_y.shape[1]
#         n_features = train_x.shape[2]

#         # Define hyperparameters
#         lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
#         learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
#         batch_size = hp.Choice('batch_size', values=[8, 16, 32])

#         # Model architecture
#         seq2seq_model = Sequential()
#         seq2seq_model.add(LSTM(lstm_units, activation='relu', input_shape=(n_steps_in, n_features)))
#         seq2seq_model.add(RepeatVector(n_steps_out))
#         seq2seq_model.add(LSTM(lstm_units, activation='relu', return_sequences=True))
#         seq2seq_model.add(TimeDistributed(Dense(1)))

#         # Compile the model
#         seq2seq_model.compile(
#             loss=tf.losses.MeanSquaredError(),
#             optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
#             metrics=[tf.metrics.RootMeanSquaredError()]
#         )

#         return seq2seq_model
    
#     def HypertuneAutoregressiveVanillaLSTM(hp):
#         """
#         Build a tunable Autoregressive Vanilla LSTM Model.
#         """
#         predictions = []

#         # Hyperparameters
#         lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
#         dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
#         dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
#         learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

#         # Input shape
#         input_ts = Input(shape=(train_x.shape[1], train_x.shape[2]))

#         # Initial LSTM Layer
#         lstm_1, *state = LSTM(lstm_units, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_ts)
#         dropout_1 = Dropout(dropout_rate)(lstm_1)

#         # Define LSTM Cell and Dense layers
#         lstm_cell = LSTMCell(lstm_units)
#         dense_1 = Dense(dense_units, activation='relu')
#         dense_2 = Dense(1)

#         # First Prediction
#         x = dense_1(dropout_1)
#         prediction = dense_2(x)
#         predictions.append(prediction)

#         # Autoregressive Predictions
#         for n in range(1, 3):
#             x = prediction
#             x, state = lstm_cell(x, states=state)
#             x = dense_1(x)
#             prediction = dense_2(x)
#             predictions.append(prediction)

#         predictions = StackAndTransposeLayer()(predictions)

#         # Build Model
#         ar_model = Model(inputs=input_ts, outputs=predictions)

#         # Compile Model
#         ar_model.compile(
#             loss=tf.losses.MeanSquaredError(),
#             optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
#             metrics=[tf.metrics.RootMeanSquaredError()]
#         )

#         return ar_model

#     if type == "SingleShot":
#         return HypertuneSingleShotVanillaLSTM
#     elif type == "Seq2Seq":
#         return HypertuneSeq2SeqVanillaLSTM
#     else:
#         return HypertuneAutoregressiveVanillaLSTM