import numpy as np

def Qinitialization(N, L):
    """
    初始化 Qbit 随机矩阵
    :param N: 搜索代理数量
    :param L: 维度
    :return: Qbit: 随机初始化的量子比特数组
    """
    # 初始化Qbit矩阵的两个分量
    Qbit = np.zeros((N, L, 2))

    # 生成范围在 [-1, 1] 的随机数
    Qbit[:, :, 0] = -1 + 2 * np.random.rand(N, L)
    Qbit[:, :, 1] = -1 + 2 * np.random.rand(N, L)

    # 计算 Qbit 的振幅 |alpha|^2 + |beta|^2 = 1，归一化
    AA = np.sqrt(Qbit[:, :, 0]**2 + Qbit[:, :, 1]**2)
    Qbit[:, :, 0] = Qbit[:, :, 0] / AA
    Qbit[:, :, 1] = Qbit[:, :, 1] / AA

    return Qbit
