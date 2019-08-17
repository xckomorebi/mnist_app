import tensorflow as tf
import numpy as np
from PIL import Image

def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))#变为灰度图
    threshold = 50#阈值，将图片二值化操作
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]#进行反色处理
            if(im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else: pass


    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)#类型转换
    img_ready = np.multiply(nm_arr, 1.0/255.0)#把值变为0~1之间的数值

    return img_ready


def getPred(img):
    # Create the model
    test = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    test_pred = tf.matmul(test, W) + b
    test_label = tf.argmax(test_pred, 1)
    
    # Load the training data
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, "./Model/model.ckpt")
    return sess.run(test_label,feed_dict={test: img})
