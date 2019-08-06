import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(parentPath)

import numpy as np
import matplotlib.pyplot as plt
from app.cnn_utils import convert_to_one_hot, load_dataset
from app.model import normalize, normalize1, model1, model, splitDataToTrainAndTest
import tensorflow as tf
from config.APP import model_path, images_path_train, resource_path, images_path_test
from util.Model import getModelName
# import os
from util.redis_client import trainIdGenerator
import logging
from util.img_loader import appendALine
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


def imgsToDataSet(img_dir):
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
        lambd = graph.get_tensor_by_name("lambd:0")
        dropout = graph.get_tensor_by_name("dropout:0")
        predict_op = graph.get_tensor_by_name("predict_op:0")
        return predict_op.eval({X: x, lambd: 0, dropout: 0})

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


def train(starter_learning_rate, learning_rate_decay, dropout_rate,num_epochs=200, minibatch_size=64, lambd=0):
    # load dataset
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train_orig, Y_train_orig = imgsToDataSet(images_path_train)
    m_train = Y_train_orig.shape[1]
    X_test_orig, Y_test_orig = imgsToDataSet(images_path_test)
    m_test = Y_test_orig.shape[1]
    # normalize x
    X_train, X_test = normalize(X_train_orig, X_test_orig)

    # convert y to one hot
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # train
    train_accuracy, test_accuracy, parameters, costs = model(
        X_train, Y_train, X_test, Y_test,
        starter_learning_rate=starter_learning_rate, learning_rate_decay=learning_rate_decay, dropout_rate=dropout_rate,
        num_epochs=num_epochs, minibatch_size=minibatch_size, lambd_train=lambd)
    # save the model

    # print accuracy
    logger.info("Train Accuracy: %f" % train_accuracy)
    logger.info("Test Accuracy: %f" % test_accuracy)
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(starter_learning_rate))
    # plt.show()
    return train_accuracy, test_accuracy, np.squeeze(costs), m_train, m_test


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
    # x, y = imgsToTrainSet(images_path_train)
    # x_test = x[0:1,:,:,:]
    # X = tf.placeholder(name='X', shape=(None, 64, 64, 3), dtype=tf.float32)
    # lambd = tf.placeholder(name='lambd', dtype=tf.float32)
    # print(lambd)
    # z = tf.cond(lambd,lambda: 5,lambda: 4)
    # W1 = tf.get_variable(name='W1', dtype=tf.float32, shape=(8, 8, 3, 8),
    #                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Z1 = tf.nn.conv2d(input=X, filter=W1, strides=(1, 1, 1, 1), padding='SAME')
    # P1 = tf.nn.max_pool(value=Z1, ksize=(1, 8, 8, 1), strides=(1, 8, 8, 1), padding='SAME')

    # with tf.Session() as sess:

    #
    #     # Run the initialization
    #     sess.run(tf.global_variables_initializer())
    #     z = sess.run(P1,feed_dict={X:x_test})
    #     print(z.shape)
    #     print(sess.run(z, feed_dict={lambd: 1}))

    # _, y = imgsToDataSet(images_path_train)
    # print(m)
    num_epochss = [300,400]
    starter_learning_rates = [0.006]
    learning_rate_decays = [0.9]
    minibatch_sizes = [64]
    lambds = [0]
    dropout_rates = [0]
    for num_epochs in num_epochss:
        for starter_learning_rate in starter_learning_rates:
            for learning_rate_decay in learning_rate_decays:
                for minibatch_size in minibatch_sizes:
                    for lambd in lambds:
                        for dropout_rate in dropout_rates:
                            iid = trainIdGenerator.gen_id()
                            train_accuracy, test_accuracy, costs, m_train, m_test = train(
                                starter_learning_rate=starter_learning_rate,
                                learning_rate_decay=learning_rate_decay,
                                dropout_rate=dropout_rate,
                                num_epochs=num_epochs,
                                minibatch_size=minibatch_size,
                                lambd=lambd
                            )
                            result = 'id:' + str(iid) +\
                                     '\tnum_epochs: ' + str(num_epochs) + \
                                     '\tstarter_learning_rate: ' + str(starter_learning_rate) + \
                                     '\tlearning_rate_decay: ' + str(learning_rate_decay) + \
                                     '\tminibatch_size: ' + str(minibatch_size) + \
                                     '\tlambd: ' + str(lambd) + \
                                     '\tdropout_rate: ' + str(dropout_rate) + \
                                     '\ttrain_accuracy: ' + str(train_accuracy) + \
                                     '\ttest_accuracy: ' + str(test_accuracy) + \
                                     '\tcost: ' + str(costs[-1]) + \
                                     '\tm_train: ' + str(m_train) + \
                                     '\tm_test: ' + str(m_test)
                            appendALine(result, path=resource_path + "train.txt")
                            cost_str = 'id:' + str(iid) +\
                                       '\tcosts: ' + str(costs)
                            appendALine(cost_str, path=resource_path + "costs.txt")



    # retrain(x, y)
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