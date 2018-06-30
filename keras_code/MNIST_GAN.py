"""
MNIST GAN implementation in keras
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Reshape, Flatten, Input, LeakyReLU

# parameters
noise_len = 100


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalise from -1 to +1
x_train = (x_train - 127.5) / 127.5


def create_generator(noise_len=100):
    model = Sequential()

    model.add(Dense(256, input_shape=(noise_len,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(784, activation='tanh'))

    return model


def create_discriminator(input_shape=(28,28,1)):
    model = Sequential()

    model.add(Dense(512, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model


def main():

    discriminator = create_discriminator()
    generator = create_generator()

    discriminator.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics='accuracy'
    )
    generator.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    noise = Input(shape=(noise_len,))
    img = generator(noise)

    discriminator.trainable = False
    score = discriminator(img)
    combined = Model(inputs=noise, outputs=score)
    combined.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )



if __name__ == '__main__':
    main()