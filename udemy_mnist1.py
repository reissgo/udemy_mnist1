import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools

def varinfo(x):
    typestr = eval("type("+x+")")
    val = eval(x)
    print(f"Variable {x} is of type {typestr} and has value {val}")


def load_and_prep_data():
    global keras_own_custom_mnist_holder_class
    global y_train_a_single_integer, y_test_as_single_int, x_train_flattened_image, x_test_flattened_image
    global x_test_28x28, y_test_28x28

    keras_own_custom_mnist_holder_class = tf.keras.datasets.mnist

    (x_train_28x28, y_train_a_single_integer), (x_test_28x28, y_test_as_single_int) = keras_own_custom_mnist_holder_class.load_data()

    # help(keras_own_custom_mnist_holder_class.load_data)

    print("Pre reshape", x_train_28x28.shape)  # (60000, 28, 28)

    x_train_flattened_image = x_train_28x28.reshape(-1, 28 * 28)
    x_train_flattened_image = x_train_flattened_image / 255.0
    x_test_flattened_image = x_test_28x28.reshape(-1, 28 * 28)
    x_test_flattened_image = x_test_flattened_image / 255.0

    print("post reshape, x_train.shape = ", x_train_flattened_image.shape)
    print("x_train[0] = ...\n", x_train_flattened_image[0])

    varinfo("y_train_a_single_integer[0]")
    print(y_train_a_single_integer.shape)
    print(y_train_a_single_integer[:15])

    y_train_split_into_10 = np.zeros((y_train_a_single_integer.shape[0], 10))

    for i in range(y_train_a_single_integer.shape[0]):
        y_train_split_into_10[i][y_train_a_single_integer[i]] = 1.0

    y_test_split_into_10 = np.zeros((y_test_as_single_int.shape[0], 10))

    for i in range(y_test_as_single_int.shape[0]):
        y_test_split_into_10[i][y_test_as_single_int[i]] = 1.0

    print("y_train_split_into_10.shape = ", y_train_split_into_10.shape)
    print("y_train_split_into_10[:5] = ...\n", y_train_split_into_10[:5])


def build_net_and_define_learning_procedure():
    global model
    model = tf.keras.Sequential([
                        tf.keras.layers.Input(28*28),
                        tf.keras.layers.Dense(50, activation="relu"),
                        tf.keras.layers.Dense(10, activation="softmax")  # sigmoid?? softmax !
                        ])

    learnrate_term = 0.001
    momentum_term = 0.9

    # list of optimisers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

    model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term, momentum_term),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term,
    #               momentum_term),
    #               loss="mse")

    print("compile complete")


def train_model():
    global training_progress_history
    # training_progress_history = model.fit(x_train, y_train_split_into_10,
    #                                       validation_data=(x_test, y_test_split_into_10),
    #                                       epochs=10)

    training_progress_history = model.fit(x_train_flattened_image, y_train_a_single_integer,
                                          validation_data=(x_test_flattened_image, y_test_as_single_int),
                                          epochs=10)


def display_learning_progress_stats():
    plt.plot(training_progress_history.history["loss"], label="This is the loss")
    plt.plot(training_progress_history.history["val_loss"], label="This is the val_loss")
    plt.legend()
    plt.show()


def plot_confusion_matrix(conmat, number_of_classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    if normalize:
        conmat = conmat.astype('float') / conmat.sum(axis=1)[:, np.newaxis]
        print("normalizing!")
    print(conmat)
    plt.imshow(conmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(number_of_classes))
    plt.xticks(tick_marks, number_of_classes, rotation=45)
    plt.yticks(tick_marks, number_of_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conmat.max() / 2.
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


load_and_prep_data()

if False:
    print("Training model from scratch")
    build_net_and_define_learning_procedure()
    train_model()
    model.save('model.h5')  # h5 appears short for hdf5 which AFAICT is the version 5 of the HDF format - see www.hdfgroup.org
    display_learning_progress_stats()
else:
    print("Loading pre-existing model")
    model = tf.keras.models.load_model('model.h5')

print("now evaluate...")
print(model.evaluate(x_test_flattened_image, y_test_as_single_int))
print("Evaluation complete.")

# correct_answer_as_single_int = model.predict(x_test).argmax()

predicted_answer_as_10_floats = model.predict(x_test_flattened_image)
predicted_answer_as_single_int = predicted_answer_as_10_floats.argmax(axis=1)

cm = confusion_matrix(y_test_as_single_int, predicted_answer_as_single_int)

#plot_confusion_matrix(cm, list(range(10)), normalize=False)

# find misclassified examples

list_of_misses = np.where(y_test_as_single_int != predicted_answer_as_single_int)[0]

random_miss_idx = np.random.choice(list_of_misses)

sz = 5
for x in range(sz):
    for y in range(sz):
        idx = x + sz * y
        plt.subplot(sz,sz,idx+1)
        plt.title("True: "+str(y_test_as_single_int[list_of_misses[idx]])+" vs Est: "+str(predicted_answer_as_single_int[list_of_misses[idx]]))
        plt.imshow(x_test_28x28[list_of_misses[idx]],cmap="gray")

plt.show()