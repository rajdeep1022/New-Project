from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_shape):

    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(3, activation='softmax'))

    return model