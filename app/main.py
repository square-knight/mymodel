
import numpy as np
import matplotlib.pyplot as plt
from app.cnn_utils import convert_to_one_hot, load_dataset
from app.model import normalize, normalize1, model1, model, splitDataToTrainAndTest
import tensorflow as tf
from config.APP import model_path, images_path_train
import os
import logging
logger = logging.getLogger(__name__)
"""
# from app.model import *

import scipy
import scipy.misc

"""

def readImageFromDisk(path):
    """
    readImageFromDisk

    :param path: str
    image path

    :return:
    ndarray of shape(64, 64, 3)
    """
    image = np.array(plt.imread(path))
    # my_image = scipy.misc.imresize(image, size=(64, 64))
    # plt.imshow(image)
    # plt.show()
    return image


def imgsToTrainSet(img_dir):
    x = []
    y = []
    if os.path.isdir(img_dir):
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = readImageFromDisk(img_path)
            group = img_name[0: 1]
            x.append(img)
            y.append(int(group))
    else:
        logger.error("is not path")
    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)[np.newaxis, :]
    return x, y


# readImageFromDisk("/Users/doom/Documents/0b0ae651-4dd0-4590-afc9-a4077da14bc7cat1_2.jpeg")

def train(x, y):
    # load dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = splitDataToTrainAndTest(x, y)
    # normalize x
    X_train, X_test = normalize(X_train_orig, X_test_orig)

    # convert y to one hot
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # hyperparameters
    learning_rate = 0.003
    num_epochs = 150
    minibatch_size = 64
    lambd = None #0.03-0.9-0.85;0.01-0.92-0.89;0.006-0.96-0.86;0.003-0.98-0.89;0.002-0.978-0.875;0.001-0.98-0.9;none-0.98-0.89
    # train
    train_accuracy, test_accuracy, parameters, costs = model(
        X_train, Y_train, X_test, Y_test,
        learning_rate=learning_rate, num_epochs=num_epochs, minibatch_size=minibatch_size, lambd=lambd)
    # save the model

    # print accuracy
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def retrain(x, y, model_identifier=None):
    # x = normalize1(x_orig)
    # y = convert_to_one_hot(y_orig, 6).T

    # hyperparameters
    learning_rate = 0.0003
    num_epochs = 100
    minibatch_size = 64
    lambd = 0.003

    train_accuracy = 0.
    test_accuracy = 0.
    parameters = {}
    costs = []
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = splitDataToTrainAndTest(x, y)
    X_train, X_test = normalize(X_train_orig, X_test_orig)
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    with tf.Session() as sess:
        model_name = getModelName(model_identifier)
        saver = tf.train.import_meta_graph(model_path + model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        # train
        train_accuracy, test_accuracy, parameters, costs = model1(
            X_train, Y_train, X_test, Y_test, sess, model_name,
            learning_rate=learning_rate, num_epochs=num_epochs, minibatch_size=minibatch_size, lambd=lambd)
    # print accuracy
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict(x, model_identifier=None):
    """
    predit number of fingers in the picture

    Arguments:
    x -- predict input, of shape(None, 64, 64, 3)
    :return:
    y -- predict result
    """

    x = normalize1(x)
    with tf.Session() as sess:
        model_name = getModelName(model_identifier)
        saver = tf.train.import_meta_graph(model_path + model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        predict_op = graph.get_tensor_by_name("predict_op:0")

        return predict_op.eval({X: x})


def getModelName(model_identifier=None):
    model_name = 'finger-model'
    if model_identifier is not None:
        model_name = model_name + '-' + model_identifier
    return model_name


# train()
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# X_train, X_test = normalize(X_train_orig,X_test_orig)
#
# img1 = readImageFromDisk("/Users/doom/Documents/d5c94724-78ce-4b63-bab2-304ee3323c4dcat1_2.jpeg")
# img2 = readImageFromDisk("/Users/doom/Documents/c171ca3b-978e-4888-ae88-28507c97996ecat1_2.jpeg")
# img = np.reshape(img2, (1, 64, 64, 3))
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(X_test_orig[3])
# plt.subplot(2,2,2)
# plt.imshow(X_test_orig[4])
# plt.subplot(2,2,3)
# plt.imshow(img1)
# plt.subplot(2,2,4)
# plt.imshow(img2)
# plt.show()
# y_predict = predict(X_test)
# print("y_predict:\n", y_predict)


def test():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        P2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        P2 = tf.contrib.layers.flatten(inputs=P2)
        print(P2.eval())
        z = tf.contrib.layers.fully_connected(P2, 2, activation_fn=None)
        print(z.eval())


if __name__ == '__main__':
    x, y = imgsToTrainSet(images_path_train)
    # train(x, y)
    retrain(x, y)
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # print(X_train_orig.shape)
    # print(Y_train_orig.shape)
    #
    # y_predict = predict(X_test_orig)
    #
    # print(X_test_orig.shape)
    # print("y_predict:\n", y_predict)




    # print(x.shape)
    # print(y.shape)
    # print(y[0, 105:107])
    # print(y[0, 28])
    # print(Y_test_orig[0,[13, 28, 105, 106]])
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(X_test_orig[13])
    # plt.subplot(2,2,2)
    # plt.imshow(X_test_orig[28])
    # plt.subplot(2,2,3)
    # plt.imshow(X_test_orig[105])
    # plt.subplot(2,2,4)
    # plt.imshow(X_test_orig[106])
    # plt.show()
