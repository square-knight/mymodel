import os
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)


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


def appendALine(string, path):
    fp = open(path, "a", encoding="utf-8")
    fp.write(string + "\n")
    fp.close()
