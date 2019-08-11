import tensorflow as tf
from app.cnn_utils import random_mini_batches
from config.APP import model_path
import logging
logger = logging.getLogger(__name__)


def fully_connected(prev_layer, num_units, is_training):
    """
    num_units参数传递该层神经元的数量,根据prev_layer参数传入值作为该层输入创建全联接神经网络
    :param prev_layer: Tensor
    该网络层输入
    :param num_units: int
    该网络层节点数
    :param is_training: bool of Tensor
    表示该网络是否处于训练状态，
    告知Batch Normalization层是否应该更新或者使用均值和方差的分布信息
    :return: Tensor
    一个新的全连接神经网络层
    """
    layer = tf.keras.layers.Dense(units=num_units,activation=None,use_bias=False)(inputs=prev_layer)
    layer = tf.keras.layers.BatchNormalization()(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


"""
向生成卷积层的'conv_layer'函数中添加Batch Normalization，我们需要以下步骤：
1.在函数声明中添加'is_training'参数，以确保可以向Batch Normalization层中传递信息
2.去除conv2d层中bias偏置属性和激活函数
3.使用'tf.keras.layers.BatchNormalization'来标准化卷积层的输出，注意，将'is_training'传递给该层，
    以确保网络适时更新数据集均值和方差统计信息。
4.将经过Batch Normalization后的值传递到ReLU激活函数中
PS:和'fully_connected'函数比较，你会发现如果你使用tf.layers包函数对全连接层进行BN操作和对卷积层进行
    BN操作没有任何区别，但是如果使用tf.nn包中的函数实现BN会发现一些小的变动
"""
"""
我们会运用以下的方法来构建神经网络的卷积层，这个卷积层很基本，我们总是使用3x3内核，ReLU激活函数
在具有奇数深度的涂层上步长为1x1，在具有偶数深度的涂层上步长为2x2。在这个网络中，我们并不打算使用池化层。
PS:该版本的函数包括批量标准化操作
"""


# def conv_layer(prev_layer, layer_depth, is_training):
#     """
#     使用给定的参数作为输入创建卷积层
#     :param prev_layer: Tensor
#     卷积输入数据
#     :param layer_depth: int
#     我们将根据网络中的图层的深度设置特征图的步长和数量
#     这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
#     :param is_training: bool of Tensor
#     表示该网络是否处于训练状态，
#     告知Batch Normalization层是否应该更新或者使用均值和方差的分布信息
#     :return: Tensor
#     一个新的卷积层
#     """
#     strides = 2 if layer_depth % 3 == 0 else 1
#     conv_layer = tf.layers.conv2d(prev_layer, layer_depth * 4, 3, strides, 'same',
#                                   use_bias=False, activation=None)
#     conv_layer = tf.keras.layers.BatchNormalization()(conv_layer, training=is_training)
#     conv_layer = tf.nn.relu(conv_layer)
#     return conv_layer

def conv_layer(prev_layer: tf.Tensor, layer_depth: int, kernel_size: int, is_training: tf.Tensor) -> tf.Tensor:
    """
    使用给定的参数作为输入创建卷积层
    :param prev_layer: Tensor
    卷积输入数据
    :param layer_depth: int
    我们将根据网络中的图层的深度设置特征图的步长和数量
    这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
    :param kernel_size: int
    卷积核大小
    :param is_training: bool of Tensor
    表示该网络是否处于训练状态，
    告知Batch Normalization层是否应该更新或者使用均值和方差的分布信息
    :return: Tensor
    一个新的卷积层
    """
    conv_layer = tf.keras.layers.Conv2D(filters=layer_depth, kernel_size=kernel_size,
                                        padding='same', activation=None
                                        , use_bias=False)(prev_layer)
    conv_layer = tf.keras.layers.BatchNormalization(axis=3)(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer


"""
批量标准化是一个新的概念，研究人员仍在尝试如何更好的使用它
一般来说，可以删除层中的偏移值(因为BN已经有了缩放和移位的功能参数)，并且在非线性激活函数之前添加BN。
然而，对于某些网络来说，使用其它的方法也能得到不错的效果

为了演示这一点，以下三个版本的conv_layer展示了实现BN的其他方法
如果你尝试使用这些函数的任何一个版本，他们都会运行良好
（尽管他们可能具体表现不一）
"""

# 在卷积层中使用偏置user_bais=True,在ReLU激活函数之前仍然添加了BN
# def conv_layer(prev_layer, layer_num, is_training):
#     strides = 2 if layer_num % 3 == 0 else 1
#     conv_layer = tf.layers.conv2d(prev_layer, layer_num * 4, 3, strides, 'same',
#                                   use_bias=True, activation=None)
#     conv_layer = tf.keras.layers.BatchNormalization()(conv_layer, training=is_training)
#     conv_layer = tf.nn.relu(conv_layer)
#     return conv_layer

# 在卷积层中使用偏置use_bias=True, 先使用ReLU激活函数处理然后添加了BN
# def conv_layer(prev_layer, layer_num, is_training):
#     strides = 2 if layer_num % 3 == 0 else 1
#     conv_layer = tf.layers.conv2d(prev_layer, layer_num * 4, 3, strides, 'same',
#                                   use_bias=True, activation=tf.nn.relu)
#     conv_layer = tf.keras.layers.BatchNormalization()(conv_layer, training=is_training)
#     return conv_layer

# 在卷积层中不使用偏置use_bias=False, 先使用ReLU激活函数处理然后添加了BN
# def conv_layer(prev_layer, layer_num, is_training):
#     strides = 2 if layer_num % 3 == 0 else 1
#     conv_layer = tf.layers.conv2d(prev_layer, layer_num * 4, 3, strides, 'same',
#                                   use_bias=False, activation=tf.nn.relu)
#     conv_layer = tf.keras.layers.BatchNormalization()(conv_layer, training=is_training)
#     return conv_layer


"""
为了修改训练函数，我们需要做以下工作：
1.Added is_training, a placeholder to store a boolean value indicating whether or not the network is training.
2.Passed is_training to te conv_layer and fully_connected functions.
3.Each time we call run on the session, we added to feed_dict the approprate value for is_training.
4.Moved the creation of train_opt inside a with tf.control_dependencies... statement.
This is necessary to get the normaization layers created with tf.layers.batch_normalization to update 
their population statistics,which we need when performing inference.
"""


def train(x_train, y_train, x_test, y_test, num_epochs, minibatch_size, starter_learning_rate, learning_rate_decay=1.):
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 0  # to keep results consistent (numpy seed)
    (m, h, w, c) = x_train.shape

    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 3], name='inputs')
    labels = tf.placeholder(tf.float32, [None, 6])

    # Add placeHolder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Feed the inputs into a series of convolutional layers
    c1 = conv_layer(inputs, 16, 4, is_training)
    p1 = tf.nn.max_pool(value=c1, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')
    c2 = conv_layer(p1, 32, 4, is_training)
    p2 = tf.nn.max_pool(value=c2, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')
    # Flatten the output from the convolutional layers
    p2 = tf.keras.layers.Flatten()(inputs=p2)
    # Add one fully connected layer
    fc1 = fully_connected(p2, 32, is_training)
    # Create the output layer with 1 node for each label
    logits = tf.keras.layers.Dense(units=6, activation=None, use_bias=False)(fc1)
    # Define loss and training operations
    model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    # Tell Tensorflow to update the population statistics while training
    # 从 ops 中过滤出 updates_ops，然后添加到指定的 collection
    ops = tf.get_default_graph().get_operations()
    update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   20000, learning_rate_decay, staircase=True)
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss, global_step=global_step)

    # Create operations to test accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1, name='prediction'), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # model saver
    saver = tf.train.Saver()

    costs = []
    # Train and test the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_x, minibatch_y) = minibatch
                _, temp_cost, step, lrate = sess.run([train_opt, model_loss, global_step, learning_rate],
                                        {inputs: minibatch_x, labels: minibatch_y, is_training: True})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 5 == 0:
                logger.info("LR:%f,Cost after epoch %i(step:%i)\t: %f" % (lrate, epoch, step, minibatch_cost))
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        # save the model
        saver.save(sess, model_path + "finger-model-bn")
        # At the end, score the final accuracy for train and test set
        acc_train = sess.run(accuracy, {inputs: x_train,
                                        labels: y_train,
                                        is_training: False})
        print('Final train accuracy: {:>3.5f}'.format(acc_train))
        acc_test = sess.run(accuracy, {inputs: x_test,
                                       labels: y_test,
                                       is_training: False})
        print('Final test accuracy: {:>3.5f}'.format(acc_test))

        return acc_train, acc_test, costs


# if __name__ == '__main__':
#     X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#     x_train = X_train_orig / 255
#     x_test = X_test_orig / 255
#     y_train = convert_to_one_hot(Y_train_orig, 6).T
#     y_test = convert_to_one_hot(Y_test_orig, 6).T
#     train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
#           num_epochs=50, minibatch_size=64, starter_learning_rate=0.006, learning_rate_decay=0.9)
