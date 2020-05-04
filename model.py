"""
This is the Main Keras Model File.
The idea and most of the work in this file is based on the Paper.
https://arxiv.org/pdf/1604.07316v1.pdf
"""

# First Let's import all the helper libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from utils import INPUT_SHAPE, batch_generator

# We are going to make a Simple Sequential Model as described in the Paper.
from keras.models import Sequential
# ModelCheckpoint will help us to save the model after every epoch if it is better than previous one.
from keras.callbacks import ModelCheckpoint
# Now we will import all the layers that we will use in our Model.
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

from keras.callbacks import TensorBoard
from time import time

# Path to our data directory
DATA_DIR = 'data'
# Percentage of test data that we want to create out of whole training data
TEST_SIZE = 0.2
# Batch size to use while training our Model
BATCH_SIZE = 50
# No of Epochs to train the model
NO_OF_EPOCHS = 5
# No of Samples to take per Epoch
SAMPLES_PER_EPOCH = 25000
# Learning Rate
LEARNING_RATE = 1.0e-4


def load_data():

    df = pd.read_csv(os.path.join(DATA_DIR, 'driving_log.csv'))

    # print(df.head())

    X = df[['center', 'left', 'right']].values
    Y = df['steering'].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=True)

    return X_train, X_valid, Y_train, Y_valid


def create_model():
    """
    The Model Architecture is pretty much the same that is described in the paper itself.
    I have just modified it a bit to perform better and quicker.
    """
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))

    # Now we are going to add some Convulation Layers identical to paper

    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(BatchNormalization())     
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(BatchNormalization()) 
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # And now finally we will Flatten our layers and eventually use Fully Connected Layers to reduce features.

    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model


def train_model(model, X_train, X_valid, Y_train, Y_valid):
    """
    This function will be called to train our model
    """

    # Let's CreateCheckPoint so as to save our best models and model Logs after every epoch.
    # This checkPoint function will be called as callback functions after every epoch.

    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    tensorboard = TensorBoard(log_dir="log\{}".format(time()))    

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

    model.fit_generator(batch_generator(DATA_DIR, X_train, Y_train, BATCH_SIZE, True),
                        SAMPLES_PER_EPOCH,
                        NO_OF_EPOCHS,
                        max_q_size=1,
                        validation_data=batch_generator(
                            DATA_DIR, X_valid, Y_valid, BATCH_SIZE, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint, tensorboard],
                        verbose=1
                        )


def main():
    data = load_data()
    model = create_model()
    train_model(model, *data)


if __name__ == '__main__':
    main()
