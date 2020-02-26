import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


keras_own_custom_mnist_holder_class = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = keras_own_custom_mnist_holder_class.load_data()

# help(keras_own_custom_mnist_holder_class.load_data)

print("Pre reshape", x_train.shape)  # (60000, 28, 28)

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype(float)

print("post reshape, x_train.shape = ", x_train.shape)
print("x_train[0] = ...\n", x_train[0])

print(y_train.shape)
print(y_train[:15])

y_train2 = np.zeros((y_train.shape[0],10))

for i in range(y_train.shape[0]):
    y_train2[i][y_train[i]] = 1

print("y_train2.shape = ", y_train2.shape)
print("y_train2[:5] = ...\n",y_train2[:5])


model = tf.keras.Sequential([
                    tf.keras.layers.Input(28*28),
                    tf.keras.layers.Dense(50, activation="relu"),
                    tf.keras.layers.Dense(10, activation="sigmoid")
                    ])

learnrate_term = 0.001
momentum_term = 0.9

#model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term, momentum_term), loss="sparse_categorical_crossentropy")
model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term, momentum_term), loss="mse")

print("compile complete")

training_progress_history = model.fit(x_train, y_train2, epochs=10)

help(training_progress_history)

plt.plot(training_progress_history.history["loss"])

plt.show()
