import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


def my_nn(input_dim):
    model = Sequential()

    model.add(Dense(4, input_dim=input_dim))

    # model.add(Dropout(0.1))

    model.add(Dense(3))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    df = pd.read_csv('data/forest.csv')

    le_month = LabelEncoder()
    le_day = LabelEncoder()

    # Преобразование текстовых данных для обучения
    le_month.fit(df.month.unique())
    le_day.fit(df.day.unique())

    sc = MinMaxScaler()

    columns = ['X', 'Y', 'FFMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

    months = le_month.transform(df.month)
    days = le_day.transform(df.day)

    df.drop('month', axis=1, inplace=True)
    df.drop('day', axis=1, inplace=True)

    df['month'] = months
    df['day'] = days
    # Скейлинг данных для лучшего обучения
    tf_col = sc.fit_transform(df[columns])
    df_t = pd.DataFrame(columns=columns, data=tf_col)

    y = df_t['area']
    df_t.drop(columns=['area'], inplace=True)

    X = df_t.values

    kf = KFold(n_splits=10)
    # model = my_nn(df_t.shape[1])
    # model = LinearRegression()
    model = SGDRegressor()
    i = 1
    mse_list = list()

    for train_index, test_index in kf.split(df_t):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
        # checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
        # history = model.fit(x=X_train, y=y_train, epochs=12, validation_data=(X_test, y_test),
        #                     callbacks=[reduce_lr, checkpointer],verbose=0)
        # history_plot(history)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        mse_ = mse(y_test, predicted)
        mse_list.append(mse_)
        print("Fold " + str(i) + " " + str())
        i += 1

    mse_list = np.array(mse_list).reshape(1, -1)
    norm_mse = sc.inverse_transform(mse_list)
    print(norm_mse)
    print(sum(sum(norm_mse)))
