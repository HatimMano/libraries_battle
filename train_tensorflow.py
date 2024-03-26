import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Charger les données MNIST
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train / 255.0
