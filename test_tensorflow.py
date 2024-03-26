import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Load the trained TensorFlow model
model = tf.keras.models.load_model('mnist_cnn_model_tf.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test, verbose=2)
print('\nTest accuracy (TensorFlow model):', test_acc)
