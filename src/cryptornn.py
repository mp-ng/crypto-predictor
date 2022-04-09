import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SECTION: PARAMETERS

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 1
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))}"


# SECTION: FUNCTIONS - classify, preprocess_df, balance


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df, train=True):
    df = df.drop(columns="future")

    # Normalization
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df[col] = preprocessing.scale(df[col].values)

    # Clean N/A values
    df.dropna(inplace=True)

    # Generate sequential data
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for col_v in df.values:
        prev_days.append(col_v[:-1])  # row of 8 values in the specified column index
        if len(prev_days) == SEQ_LEN:  # after 60 days of data is collected
            sequential_data.append([np.array(prev_days), col_v[-1]])

    np.random.shuffle(sequential_data)

    # Balance the training set
    if train:
        balance(sequential_data)

    # Train-validation split
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y


def balance(data):
    buys = []
    sells = []
    for seq, target in data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    data = buys + sells
    np.random.shuffle(data)


# SECTION: DATA PREPROCESSING

main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    # Name columns
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    # Index data
    df.set_index("time", inplace=True)

    # Extract columns
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    # Merge Data
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

# SECTION: TRAIN-VALIDATION SPLIT

times = main_df.index.values

last_5pct = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df, True)
validation_x, validation_y = preprocess_df(validation_main_df, False)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

print(f"Train data: {len(train_x)} | validation: {len(validation_x)}")
print(f"Don't buys: {np.count_nonzero(train_y == 0)} | buys: {np.count_nonzero(train_y == 1)}")
print(f"VALIDATION Don't buys: {np.count_nonzero(validation_y == 0)} | buys: {np.count_nonzero(validation_y == 1)}")

# SECTION: BUILD MODEL

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)

# SECTION: COMPILE MODEL

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"
checkpoint = ModelCheckpoint("models/{}_{}.model".format(NAME, filepath),
                             monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

# SECTION: TRAIN MODEL

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint]
)
