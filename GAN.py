# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:16:52 2022

@author: mmqq2
"""

from keras.models import Model, Sequential
from keras.layers import Dense, Input
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = (x_test.astype(np.float32) - 127.5)/127.5
mnist_data=x_test.reshape(10000, 784)
print(mnist_data.shape)
len(mnist_data)

# 생성자 생성
def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=784, activation='tanh'))
    return generator

g = create_generator()
g.summary()

# 판별자 생성
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=512,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.2, beta_1=0.5))
    return discriminator

d = create_discriminator()
d.summary()

# GAN 생성
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_gan(d, g)
gan.summary()

# GAN 훈련 및 결과 확인
def plot_generated_images(generator):
    noise = np.random.normal(loc=0, scale=1, size=[100, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

batch_size = 128
epochs = 1000

# GAN 최종 훈련
for e in tqdm(range(epochs)):
    noise = np.random.normal(0,1, [batch_size, 100])
    generated_images=g.predict(noise)
    image_batch=mnist_data[np.random.randint(low=0,high=mnist_data.shape[0],size=batch_size)]
    X = np.concatenate([image_batch, generated_images])
    y_dis = np.zeros(2*batch_size)
    y_dis[:batch_size] = 1

    d.trainable = True
    d.train_on_batch(X, y_dis)

    noise = np.random.normal(0,1, [batch_size, 100])
    y_gen = np.ones(batch_size)
    d.trainable = False
    gan.train_on_batch(noise, y_gen)
    
    if e == 0 or e % 1000 == 0:    
        plot_generated_images(g)
