from __future__ import print_function  # 兼容 Python 2/3 的打印函数
import numpy as np  # 导入数值计算库
import random  # 随机数生成模块
from copy import deepcopy  # 深拷贝函数

# for the replication of the same simulation data, uncomment next line  # 若需复现相同数据，取消注释
# np.random.seed(2020)


""" Function to generate simulation data with two modalities """  # 生成双模态模拟数据的函数


def multi_modal_simulator(n_clusters,  # 聚类数量
                          n,  # 模拟细胞数
                          d_1,  # 转录特征维度
                          d_2,  # 形态特征维度
                          k,  # 潜在空间维度
                          sigma_1,  # 转录噪声方差
                          sigma_2,  # 形态噪声方差
                          decay_coef_1,  # 转录掉零衰减系数
                          decay_coef_2,  # 形态掉零衰减系数
                          merge_prob  # 聚类合并概率
                          ):
    """
    Generate simulated data with two modalities.

    Parameters:
      n_clusters:       number of ground truth clusters.
      n:                number of cells to simulate.
      d_1:              dimension of features for transcript modality.
      d_2:              dimension of features for morphological modality.
      k:                dimension of latent code to generate simulate data (for both modality)
      sigma_1:          variance of gaussian noise for transcript modality.
      sigma_2:          variance of gaussian noise for morphological modality.
      decay_coef_1:     decay coefficient of dropout rate for transcript modality.
      decay_coef_2:     decay coefficient of dropout rate for morphological modality.
      merge_prob:       probability to merge neighbor clusters for the generation of modality-specific
                        clusters (same for both modalities)


    Output:
      a dataframe with keys as follows

      'true_cluster':   true cell clusters, a vector of length n

      'data_a_full':    feature matrix of transcript without dropouts
      'data_a_dropout': feature matrix of transcript with dropouts
      'data_a_label':   cluster labels to generate transcript features after merging

      'data_b_full':    feature matrix of morphology without dropouts
      'data_b_dropout': feature matrix of morphology with dropouts
      'data_b_label':   cluster labels to generate morphological features after merging


    Altschuler & Wu Lab 2020.
    Software provided as is under MIT License.
    """

    # data dict for output
    data = {}  # 用于存储结果的字典

    """ generation of true cluster labels """  # 生成真实的聚类标签
    cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])  # 随机生成聚类编号
    data['true_cluster'] = cluster_ids  # 保存真实标签

    """ merge a group of true clusters randomly """  # 随机合并真实聚类
    # divide clusters into two equal groups  # 将聚类分成两组
    section_a = np.arange(np.floor(n_clusters / 2.0))  # 第一组索引
    section_b = np.arange(np.floor(n_clusters / 2.0), n_clusters)  # 第二组索引

    uniform_a = np.random.uniform(size=section_a.size - 1)  # 生成随机数用于合并
    uniform_b = np.random.uniform(size=section_b.size - 1)  # 生成随机数用于合并

    section_a_cp = section_a.copy()  # 备份数组
    section_b_cp = section_b.copy()  # 备份数组

    # randomly merge two neighbor clusters at merge_prob  # 按概率合并相邻聚类
    for i in range(uniform_a.size):
        if uniform_a[i] < merge_prob:
            section_a_cp[i + 1] = section_a_cp[i]
    for i in range(uniform_b.size):
        if uniform_b[i] < merge_prob:
            section_b_cp[i + 1] = section_b_cp[i]
    # reindex  # 重新编号聚类
    cluster_ids_a = cluster_ids.copy()  # 模态 A 聚类标签
    cluster_ids_b = cluster_ids.copy()  # 模态 B 聚类标签
    for i in range(section_a.size):
        idx = np.nonzero(cluster_ids == section_a[i])[0]
        cluster_ids_a[idx] = section_a_cp[i]
    for i in range(section_b.size):
        idx = np.nonzero(cluster_ids == section_b[i])[0]
        cluster_ids_b[idx] = section_b_cp[i]

    """ Simulation of transcriptional modality """  # 转录模态模拟
    # generate latent code  # 生成潜在表示
    Z_a = np.zeros([k, n])  # 初始化潜在矩阵
    for id in list(set(cluster_ids_a)):
        idxs = cluster_ids_a == id  # 当前聚类索引
        cluster_mu = np.random.random([k]) - 0.5  # 聚类中心
        Z_a[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()  # 生成潜在变量
    # random projection matrix  # 随机投影矩阵
    A_a = np.random.random([d_1, k]) - 0.5
    # gaussian noise  # 高斯噪声
    noise_a = np.random.normal(0, sigma_1, size=[d_1, n])
    # raw feature  # 线性变换得到特征
    X_a = np.dot(A_a, Z_a).transpose()
    X_a[X_a < 0] = 0  # 截断负值
    # dropout  # 计算丢失
    cutoff = np.exp(-decay_coef_1 * (X_a ** 2))
    X_a = X_a + noise_a.T  # 加噪声
    X_a[X_a < 0] = 0  # 再次截断
    Y_a = deepcopy(X_a)  # 拷贝用于随机失活
    rand_matrix = np.random.random(Y_a.shape)  # 随机矩阵
    zero_mask = rand_matrix < cutoff  # 失活掩码
    Y_a[zero_mask] = 0  # 应用失活

    data['data_a_full'] = X_a  # 完整的转录特征
    data['data_a_dropout'] = Y_a  # 含失活的转录特征
    data['data_a_label'] = cluster_ids_a  # 转录聚类标签

    """ Simulation of morphological modality """  # 形态模态模拟
    # generate latent code  # 生成潜在表示
    Z_b = np.zeros([k, n])  # 初始化潜在矩阵
    for id in list(set(cluster_ids_b)):
        idxs = cluster_ids_b == id  # 当前聚类索引
        cluster_mu = (np.random.random([k]) - .5)  # 聚类中心
        Z_b[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()  # 生成潜在变量

    # first layer of neural network  # 第一层网络
    A_b_1 = np.random.random([d_2, k]) - 0.5
    X_b_1 = np.dot(A_b_1, Z_b)
    X_b_1 = 1 / (1 + np.exp(-X_b_1))  # Sigmoid 激活

    # second layer of neural network  # 第二层网络
    A_b_2 = np.random.random([d_2, d_2]) - 0.5
    noise_b = np.random.normal(0, sigma_2, size=[d_2, n])
    X_b = (np.dot(A_b_2, X_b_1) + noise_b).transpose()  # 加噪声并线性变换
    X_b = 1 / (1 + np.exp(-X_b))  # Sigmoid 激活

    # random dropouts  # 随机置零
    Y_b = deepcopy(X_b)  # 拷贝特征
    rand_matrix = np.random.random(Y_b.shape)  # 随机矩阵
    zero_mask = rand_matrix < decay_coef_2  # 置零掩码
    Y_b[zero_mask] = 0  # 应用置零

    data['data_b_full'] = X_b  # 完整的形态特征
    data['data_b_dropout'] = Y_b  # 含失活的形态特征
    data['data_b_label'] = cluster_ids_b  # 形态聚类标签

    return data  # 返回模拟数据
