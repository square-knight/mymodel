from app.model import *
import scipy


def readImageFromDisk(path):
    """
    readImageFromDisk

    :param path:
    image path

    :return:
    ndarray of shape(1, 64, 64, 3)
    """
    image = np.array(plt.imread(path))
    my_image = scipy.misc.imresize(image, size=(64, 64))
    # plt.imshow(image)
    # plt.show()
    return my_image


# readImageFromDisk("/Users/doom/Documents/0b0ae651-4dd0-4590-afc9-a4077da14bc7cat1_2.jpeg")

def train():

    # load dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # normalize x
    X_train, X_test = normalize(X_train_orig,X_test_orig)

    # convert y to one hot
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # hyperparameters
    learning_rate = 0.003
    num_epochs = 200
    minibatch_size = 64
    lambd = 0.003
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


def predict(x):

    """
    predit number of fingers in the picture

    Arguments:
    x -- predict input, of shape(None, 64, 64, 3)
    :return:
    y -- predict result
    """

    x = normalize1(x)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../resource/model/finger-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../resource/model/'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        predict_op = graph.get_tensor_by_name("predict_op:0")

        return predict_op.eval({X: x})


# train()
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# X_train, X_test = normalize(X_train_orig,X_test_orig)
#
img1 = readImageFromDisk("/Users/doom/Documents/d5c94724-78ce-4b63-bab2-304ee3323c4dcat1_2.jpeg")
img2 = readImageFromDisk("/Users/doom/Documents/c171ca3b-978e-4888-ae88-28507c97996ecat1_2.jpeg")
img = np.reshape(img2, (1, 64, 64, 3))
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
y_predict = predict(img)
print("y_predict:\n", y_predict)


def test():

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        P2 = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
        P2 = tf.contrib.layers.flatten(inputs=P2)
        print(P2.eval())
        z = tf.contrib.layers.fully_connected(P2, 2, activation_fn=None)
        print(z.eval())

