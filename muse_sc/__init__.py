import numpy as np  # 导入数值计算库 NumPy
from .muse_architecture import structured_embedding  # 导入 MUSE 架构函数
from scipy.spatial.distance import pdist  # 导入计算距离的函数
import phenograph  # 导入 PhenoGraph 聚类库
import tensorflow.compat.v1 as tf  # 导入 TensorFlow 兼容模块

tf.disable_v2_behavior()  # 关闭 TensorFlow v2 行为以兼容 v1

""" Model fitting and feature prediction of MUSE """


def muse_fit_predict(data_x,  # 转录组输入数据
                     data_y,  # 形态学输入数据
                     label_x,  # 转录组初始标签
                     label_y,  # 形态学初始标签
                     latent_dim=100,  # 潜在空间维度
                     n_epochs=500,  # 训练总轮数
                     lambda_regul=5,  # 正则化权重
                     lambda_super=5):  # 自监督权重
    """
        MUSE model fitting and predicting:
          This function is used to train the MUSE model on multi-modality data

        Parameters:
          data_x:       input for transcript modality; matrix of  n * p, where n = number of cells, p = number of genes.
          data_y:       input for morphological modality; matrix of n * q, where n = number of cells, q is the feature dimension.
          label_x:      initial reference cluster label for transcriptional modality.
          label_y:      inital reference cluster label for morphological modality.
          latent_dim:   feature dimension of joint latent representation.
          n_epochs:     maximal epoch used in training.
          lambda_regul: weight for regularization term in the loss function.
          lambda_super: weight for supervised learning loss in the loss function.

        Output:
          latent:       joint latent representation learned by MUSE.
          reconstruct_x:reconstructed feature matrix corresponding to input data_x.
          reconstruct_y:reconstructed feature matrix corresponding to input data_y.
          latent_x:     modality-specific latent representation corresponding to data_x.
          latent_y:     modality-specific latent representation corresponding to data_y.

        Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
        Software provided as is under MIT License.
    """

    """ initial parameter setting """
    # parameter setting for neural network
    n_hidden = 128  # 神经网络隐藏层节点数
    learn_rate = 1e-4  # 优化器学习率
    batch_size = 64  # 每批次训练的样本数量
    n_epochs_init = 200  # 初始化阶段训练轮数
    print_epochs = 50  # 打印损失的间隔
    cluster_update_epoch = 200  # 更新模态特定聚类的周期

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]  # x 特征维度
    feature_dim_y = data_y.shape[1]  # y 特征维度
    n_sample = data_x.shape[0]  # 样本数量

    # GPU configuration
    # config = tf.ConfigProto()  # GPU 配置示例
    # config.gpu_options.allow_growth = True  # 按需分配 GPU 显存

    """ construct computation graph using TensorFlow """
    tf.reset_default_graph()  # 重置默认计算图

    # raw data from two modalities
    x = tf.placeholder(tf.float32, shape=[None, feature_dim_x], name='input_x')  # x 模态输入
    y = tf.placeholder(tf.float32, shape=[None, feature_dim_y], name='input_y')  # y 模态输入

    # labels inputted for references
    ref_label_x = tf.placeholder(tf.float32, shape=[None], name='ref_label_x')  # x 的参考标签
    ref_label_y = tf.placeholder(tf.float32, shape=[None], name='ref_label_y')  # y 的参考标签

    # hyperparameter in triplet loss
    triplet_lambda = tf.placeholder(tf.float32, name='triplet_lambda')  # 三元组损失权重
    triplet_margin = tf.placeholder(tf.float32, name='triplet_margin')  # 三元组损失间隔

    # network architecture
    z, x_hat, y_hat, encode_x, encode_y, loss, \
    reconstruction_error, weight_penalty, \
    trip_loss_x, trip_loss_y = structured_embedding(x,
                                                    y,
                                                    ref_label_x,
                                                    ref_label_y,
                                                    latent_dim,
                                                    triplet_margin,
                                                    n_hidden,
                                                    lambda_regul,
                                                    triplet_lambda)
    # optimization operator
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    print('++++++++++ MUSE for multi-modality single-cell analysis ++++++++++')
    """ MUSE optimization """
    total_batch = int(n_sample / batch_size)

    with tf.Session() as sess:

        """ initialization of autoencoder architecture for MUSE """
        print('MUSE initialization')
        # global parameter initialization
        sess.run(tf.global_variables_initializer(), feed_dict={triplet_lambda: 0,
                                                               triplet_margin: 0})

        for epoch in range(n_epochs_init):
            # randomly permute samples
            random_idx = np.random.permutation(n_sample)
            data_train_x = data_x[random_idx, :]
            data_train_y = data_y[random_idx, :]

            for i in range(total_batch):  # 遍历所有批次
                # input data batches
                offset = (i * batch_size) % (n_sample)  # 计算当前批次起始索引
                batch_x_input = data_train_x[offset:(offset + batch_size), :]  # 当前批次的 x 数据
                batch_y_input = data_train_y[offset:(offset + batch_size), :]  # 当前批次的 y 数据

                # initialize parameters without self-supervised loss (triplet_lambda=0)
                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    ref_label_x: np.zeros(batch_x_input.shape[0]),  # 占位标签
                                    ref_label_y: np.zeros(batch_y_input.shape[0]),  # 占位标签
                                    triplet_lambda: 0,
                                    triplet_margin: 0})

            # calculate and print loss terms for current epoch
            if epoch % print_epochs == 0:  # 按设定周期打印损失
                L_total, L_reconstruction, L_weight = \
                    sess.run((loss, reconstruction_error, weight_penalty),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        ref_label_x: np.zeros(data_train_x.shape[0]),  # triplet_lambda 为 0 时无作用
                                        ref_label_y: np.zeros(data_train_y.shape[0]),  # triplet_lambda 为 0 时无作用
                                        triplet_lambda: 0,
                                        triplet_margin: 0})

                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight))  # 输出当前损失

        # estimate the margin for the triplet loss
        latent, reconstruct_x, reconstruct_y = \
            sess.run((z, x_hat, y_hat),
                     feed_dict={x: data_x,
                                y: data_y,
                                ref_label_x: np.zeros(data_x.shape[0]),  # 占位标签
                                ref_label_y: np.zeros(data_y.shape[0]),  # 占位标签
                                triplet_lambda: 0,
                                triplet_margin: 0})
        latent_pd_matrix = pdist(latent, 'euclidean')  # 计算潜在空间距离
        latent_pd_sort = np.sort(latent_pd_matrix)  # 对距离进行排序
        select_top_n = np.int(latent_pd_sort.size * 0.2)  # 取前 20% 距离
        margin_estimate = np.median(latent_pd_sort[-select_top_n:]) - np.median(latent_pd_sort[:select_top_n])  # 估计间隔

        # refine MUSE parameters with reference labels and triplet losses
        for epoch in range(n_epochs_init):  # 第二阶段初始化
            # randomly permute samples
            random_idx = np.random.permutation(n_sample)  # 打乱样本顺序
            data_train_x = data_x[random_idx, :]  # 打乱后的 x 数据
            data_train_y = data_y[random_idx, :]  # 打乱后的 y 数据
            label_train_x = label_x[random_idx]  # 对应的 x 标签
            label_train_y = label_y[random_idx]  # 对应的 y 标签

            for i in range(total_batch):  # 逐批更新参数
                # data batches
                offset = (i * batch_size) % (n_sample)  # 计算批次偏移
                batch_x_input = data_train_x[offset:(offset + batch_size), :]  # 获取 x 批次
                batch_y_input = data_train_y[offset:(offset + batch_size), :]  # 获取 y 批次
                label_x_input = label_train_x[offset:(offset + batch_size)]  # 批次 x 标签
                label_y_input = label_train_y[offset:(offset + batch_size)]  # 批次 y 标签

                # refine parameters
                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    ref_label_x: label_x_input,
                                    ref_label_y: label_y_input,
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})  # 执行训练步

            # calculate loss on all input data for current epoch
            if epoch % print_epochs == 0:  # 按周期记录损失
                L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y = \
                    sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        ref_label_x: label_train_x,
                                        ref_label_y: label_train_y,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet: %03.5f,\t y triplet: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y))  # 输出初始化阶段损失

        # update cluster labels based modality-specific latents
        latent_x, latent_y = \
            sess.run((encode_x, encode_y),
                     feed_dict={x: data_x,
                                y: data_y,
                                ref_label_x: label_x,
                                ref_label_y: label_y,
                                triplet_lambda: lambda_super,
                                triplet_margin: margin_estimate})

        # update cluster labels using PhenoGraph
        label_x_update, _, _ = phenograph.cluster(latent_x)  # 根据 x 特征聚类
        label_y_update, _, _ = phenograph.cluster(latent_y)  # 根据 y 特征聚类
        print('Finish initialization of MUSE')  # 打印初始化完成

        ''' Training of MUSE '''
        for epoch in range(n_epochs):
            # randomly permute samples
            random_idx = np.random.permutation(n_sample)  # 打乱样本
            data_train_x = data_x[random_idx, :]  # 重新排列 x 数据
            data_train_y = data_y[random_idx, :]  # 重新排列 y 数据
            label_train_x = label_x_update[random_idx]  # 对应的 x 标签
            label_train_y = label_y_update[random_idx]  # 对应的 y 标签

            # loop over all batches
            for i in range(total_batch):
                # batch data
                offset = (i * batch_size) % (n_sample)  # 计算当前批次索引
                batch_x_input = data_train_x[offset:(offset + batch_size), :]  # 当前批次 x
                batch_y_input = data_train_y[offset:(offset + batch_size), :]  # 当前批次 y
                batch_label_x_input = label_train_x[offset:(offset + batch_size)]  # 当前批次 x 标签
                batch_label_y_input = label_train_y[offset:(offset + batch_size)]  # 当前批次 y 标签

                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    ref_label_x: batch_label_x_input,
                                    ref_label_y: batch_label_y_input,
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})  # 执行训练

            # calculate and print losses on whole training dataset
            if epoch % print_epochs == 0:  # 周期性打印整体损失
                L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y = \
                    sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        ref_label_x: label_train_x,
                                        ref_label_y: label_train_y,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})
                # print cost every epoch
                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet loss: %03.5f,\t y triplet loss: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y))  # 输出训练阶段损失

            # update cluster labels based on new modality-specific latent representations
            if epoch % cluster_update_epoch == 0:  # 定期更新聚类标签
                latent_x, latent_y = \
                    sess.run((encode_x, encode_y),
                             feed_dict={x: data_x,
                                        y: data_y,
                                        ref_label_x: label_x,
                                        ref_label_y: label_y,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                # use PhenoGraph to obtain cluster label
                label_x_update, _, _ = phenograph.cluster(latent_x)  # 更新 x 聚类
                label_y_update, _, _ = phenograph.cluster(latent_y)  # 更新 y 聚类

        """ MUSE output """
        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = \
            sess.run((z, x_hat, y_hat, encode_x, encode_y),
                     feed_dict={x: data_x,
                                y: data_y,
                                ref_label_x: label_x,  # no effects to representations
                                ref_label_y: label_y,  # no effects to representations
                                triplet_lambda: lambda_super,
                                triplet_margin: margin_estimate})

        print('++++++++++ MUSE completed ++++++++++')  # 打印训练完成

    return latent, reconstruct_x, reconstruct_y, latent_x, latent_y  # 返回结果
