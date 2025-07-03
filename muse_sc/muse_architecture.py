from .triplet_loss import batch_hard_triplet_loss  # 导入三元组损失函数
import tensorflow.compat.v1 as tf  # 导入 TensorFlow 兼容模块

tf.disable_v2_behavior()  # 关闭 TensorFlow v2 行为
""" model structure of MUSE """


def structured_embedding(x, y, label_x, label_y, dim_z, triplet_margin,
                         n_hidden, weight_penalty, triplet_lambda):  # 构建网络结构并计算损失
    """
    Construct structure and loss function of MUSE

    Parameters:
        x:              input batches for transcript modality; matrix of  n * p, where n = batch size, p = number of genes
        y:              input batches for morphological modality; matrix of n * q, where n = batch size, q is the feature dimension
        label_x:        input sample labels for modality x
        label_y:        input sample labels for modality y
        dim_z:          dimension of joint latent representation
        triplet_margin: margin for triplet loss
        n_hidden:       hidden node number for encoder and decoder layers
        weight_penalty: weight for sparse penalty
        triplet_lambda: weight for triplet loss

    Outputs:
        z:              joint latent representations (n * dim_z)
        x_hat:          reconstructed x (same shape as x)
        y_hat:          reconstructed y (same shape as y)
        encode_x:       latent representation for modality x
        encode_y:       latent representation for modality y
        loss:           total loss
        reconstruct_loss: reconstruction loss
        sparse_penalty: sparse penalty
        trip_loss_x:    triplet loss for x
        trip_loss_y:    triplet loss for y

    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
    Software provided as is under MIT License.
    """

    # encoder
    z, encode_x, encode_y = multiview_encoder(x, y, n_hidden, dim_z)  # 编码器输出联合和单独的表示

    # decoder
    w_init = tf.initializers.random_normal()  # 权重初始化
    w_x = tf.get_variable('w_selection_x', [z.get_shape()[1], z.get_shape()[1]], initializer=w_init)  # x 解码器选择矩阵
    w_y = tf.get_variable('w_selection_y', [z.get_shape()[1], z.get_shape()[1]], initializer=w_init)  # y 解码器选择矩阵

    with tf.variable_scope("decoder_x"):
        z_x = tf.matmul(z, w_x)  # 选择与 x 相关的特征
        x_hat = decoder(z_x, n_hidden, x.get_shape()[1])  # 解码得到重建的 x

    with tf.variable_scope("decoder_y"):
        z_y = tf.matmul(z, w_y)  # 选择与 y 相关的特征
        y_hat = decoder(z_y, n_hidden, y.get_shape()[1])  # 解码得到重建的 y

    # sparse penalty
    sparse_x = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(w_x), axis=1)))  # x 选择矩阵稀疏项
    sparse_y = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(w_y), axis=1)))  # y 选择矩阵稀疏项
    sparse_penalty = sparse_x + sparse_y  # 总稀疏惩罚

    # reconstruction errors (for x modality, only non-zero entries were used)
    x_mask = tf.sign(x)  # 获取非零掩码
    reconstruct_x = tf.reduce_sum(tf.norm(tf.multiply(x_mask, (x_hat - x)))) / tf.reduce_sum(x_mask)  # x 重建误差
    reconstruct_y = tf.reduce_mean(tf.norm(y_hat - y))  # y 重建误差
    reconstruct_loss = reconstruct_x + reconstruct_y  # 总重建误差

    # triplet errors
    trip_loss_x = batch_hard_triplet_loss(label_x, z, triplet_margin)  # x 的三元组损失
    trip_loss_y = batch_hard_triplet_loss(label_y, z, triplet_margin)  # y 的三元组损失

    loss = reconstruct_loss + weight_penalty * sparse_penalty \
           + triplet_lambda * trip_loss_x \
           + triplet_lambda * trip_loss_y  # 总损失

    return z, x_hat, y_hat, encode_x, encode_y, \
           loss, reconstruct_loss, sparse_penalty, trip_loss_x, trip_loss_y  # 返回结果


def multiview_encoder(x, y, n_hidden, dim_z):  # 联合编码器
    """
    Encoder combines x and y to a joint latent representation

    Parameters:
        x:              input batches for transcript modality; matrix of  n * p, where n = batch size, p = number of genes
        y:              input batches for morphological modality; matrix of n * q, where n = batch size, q is the feature dimension
        n_hidden:       hidden node number for encoder and decoder layers
        dim_z:          dimension of joint latent representations

    Outputs:
        latent:         joint latent representations
        h_x:            latent representation for modality x
        h_y:            latent representation for modality y

    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
    Software provided as is under MIT License.
    """

    # encoder for each modality
    with tf.variable_scope("encoder_x"):
        h_x = encoder(x, n_hidden)  # 编码 x 模态
    with tf.variable_scope("encoder_y"):
        h_y = encoder(y, n_hidden)  # 编码 y 模态

    # combine h_x and h_y to joint latent representation
    w_init = tf.keras.initializers.VarianceScaling()  # 权重初始化
    b_init = tf.constant_initializer(0.)  # 偏置初始化
    h = tf.concat([h_x, h_y], 1)  # 拼接两个模态
    wo = tf.get_variable('wo', [h.get_shape()[1], dim_z], initializer=w_init)  # 输出权重
    bo = tf.get_variable('bo', [dim_z], initializer=b_init)  # 输出偏置
    latent = tf.matmul(h, wo) + bo  # 得到联合表示

    return latent, h_x, h_y  # 返回联合及单独表示


def encoder(x, n_hidden):  # 单模态编码器
    """
    Encoder for single modality

    Parameters:
        x:              input batches of single modality
        n_hidden:       hidden node number

    Outputs:
        o:              latent representation for single modality

    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
    Software provided as is under MIT License.
    """

    # initializers
    w_init = tf.keras.initializers.VarianceScaling()  # 权重初始化
    b_init = tf.constant_initializer(0.)  # 偏置初始化

    # 1st hidden layer
    w0 = tf.get_variable('w0_e', [x.get_shape()[1], n_hidden], initializer=w_init)  # 第一层权重
    b0 = tf.get_variable('b0_e', [n_hidden], initializer=b_init)  # 第一层偏置
    h0 = tf.matmul(x, w0) + b0  # 线性变换
    h0 = tf.nn.elu(h0)  # 激活函数

    # 2nd hidden layer
    w1 = tf.get_variable('w1_e', [h0.get_shape()[1], n_hidden], initializer=w_init)  # 第二层权重
    b1 = tf.get_variable('b1_e', [n_hidden], initializer=b_init)  # 第二层偏置
    h1 = tf.matmul(h0, w1) + b1  # 第二层线性变换
    o = tf.nn.tanh(h1)  # 输出采用 tanh 激活

    return o  # 返回编码结果


def decoder(z, n_hidden, n_output):  # 单模态解码器
    """
    Decoder for single modality

    Parameters:
        z:              latent representation for single modality
        n_hidden:       hidden node number in decoder
        n_output:       feature dimension of original data

    Outputs:
        y:              reconstructed feature

    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
    Software provided as is under MIT License.
    """

    # initializers
    w_init = tf.keras.initializers.VarianceScaling()  # 权重初始化
    b_init = tf.constant_initializer(0.)  # 偏置初始化

    # 1st hidden layer
    w0 = tf.get_variable('w0_d', [z.get_shape()[1], n_hidden], initializer=w_init)  # 第一层权重
    b0 = tf.get_variable('b0_d', [n_hidden], initializer=b_init)  # 第一层偏置
    h0 = tf.matmul(z, w0) + b0  # 线性变换
    h0 = tf.nn.elu(h0)  # 激活函数

    # 2nd hidden layer
    w1 = tf.get_variable('w1_d', [h0.get_shape()[1], n_hidden], initializer=w_init)  # 第二层权重
    b1 = tf.get_variable('b1_d', [n_hidden], initializer=b_init)  # 第二层偏置
    h1 = tf.matmul(h0, w1) + b1  # 第二层线性变换
    h1 = tf.nn.tanh(h1)  # tanh 激活

    # output layer-mean
    wo = tf.get_variable('wo_d', [h1.get_shape()[1], n_output], initializer=w_init)  # 输出层权重
    bo = tf.get_variable('bo_d', [n_output], initializer=b_init)  # 输出层偏置
    y = tf.matmul(h1, wo) + bo  # 得到重建结果

    return y  # 返回重建特征
