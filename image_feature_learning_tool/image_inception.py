'''
Inference of inception-v3 model with pretrained parameters on ImageNet
'''


import tensorflow.compat.v1 as tf  # 导入 TensorFlow 兼容模块

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()  # 关闭 tf2.0 的动态执行模式
import tensorflow_hub as hub  # 导入 TensorFlow Hub 模块
import numpy as np  # 导入 NumPy 库
import cv2  # 导入 OpenCV 库
import pandas as pd  # 导入 Pandas 库

# Load saved inception-v3 model
module = hub.Module("./inception_v3")  # 加载保存的 Inception-v3 模型

# images should be resized to 299x299
input_imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])  # 定义模型输入
features = module(input_imgs)  # 计算特征

# Provide the file indices
# This can be changed to image indices in strings or other formats
spot_info = pd.read_csv('spot_info.csv', header=0, index_col=None)  # 读取图片编号
image_no = spot_info.shape[0]  # 图片总数量

with tf.Session() as sess:  # 创建 TensorFlow 会话
    sess.run(tf.global_variables_initializer())  # 初始化变量
    img_all = np.zeros([image_no, 299, 299, 3])  # 存放所有图片的数组

    # Load all images and combine them as a single matrix
    # This loop can be changed to match the provided image information
    for i in range(image_no):  # 逐个加载图片
        # Here, all images are stored in example_img and in *.npy format
        # if using image format, np.load() can be replaced by cv2.imread()
        file_name = './example_img/' + spot_info.iloc[i, 2] + '.npy'  # 构建文件名
        temp = np.load(file_name)  # 读取图像数据
        temp2 = temp.astype(np.float32) / 255.0  # 转为浮点并归一化
        img_all[i, :, :, :] = temp2  # 保存到图像矩阵中

    # Check if the image are loaded successfully.
    if (i == image_no - 1):  # 判断是否读取完毕
        print('+++Successfully load all images+++')  # 打印成功信息
    else:
        print('+++Image patches missing+++')  # 打印缺失信息

    # Input combined image matrix to Inception-v3 and output last layer as deep feature
    fea = sess.run(features, feed_dict={input_imgs: img_all})  # 计算图像特征

    # Save inferred image features
    np.save('Inception_img_feature.npy', fea)  # 保存特征到文件

  